"""
Training Engine for DattaBot.

Handles different training modes:
- PRETRAIN: Initial pretraining on large corpus
- PRETRAIN_VALIDATION: Validation during pretraining
- PRETRAIN_EVALUATION: Evaluation benchmarks during pretraining
- MIDTRAIN: Continued pretraining / domain adaptation
- POSTTRAIN: Fine-tuning / instruction tuning
- POSTTRAIN_VALIDATION: Validation during post-training
"""

import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import auto, Enum
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import tqdm
from src.agent_components import DattaBotAgentComponents
from src.agent_config import get_agent_config
from src.api_interface import DattaBotAPIResponse
from src.checkpointing import DattaBotCheckpointManager, get_checkpoint_manager
from src.logger import get_logger
from src.tokenizer import DattaBotTokenizer
from src.util import is_autocast_enabled, is_rank_0
from torch import nn, Tensor
from torch.utils.data.distributed import DistributedSampler

# TPU support
try:
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


class TrainingMode(Enum):
    """Supported training modes."""

    PRETRAIN = auto()
    PRETRAIN_VALIDATION = auto()
    PRETRAIN_EVALUATION = auto()
    MIDTRAIN = auto()
    MIDTRAIN_VALIDATION = auto()
    POSTTRAIN = auto()
    POSTTRAIN_VALIDATION = auto()

    @property
    def is_training(self) -> bool:
        return self in (
            TrainingMode.PRETRAIN,
            TrainingMode.MIDTRAIN,
            TrainingMode.POSTTRAIN,
        )

    @property
    def is_validation(self) -> bool:
        return self in (
            TrainingMode.PRETRAIN_VALIDATION,
            TrainingMode.MIDTRAIN_VALIDATION,
            TrainingMode.POSTTRAIN_VALIDATION,
        )

    @property
    def is_evaluation(self) -> bool:
        return self == TrainingMode.PRETRAIN_EVALUATION

    @property
    def metric_prefix(self) -> str:
        prefixes = {
            TrainingMode.PRETRAIN: "pretrain",
            TrainingMode.PRETRAIN_VALIDATION: "pretrain_val",
            TrainingMode.PRETRAIN_EVALUATION: "pretrain_eval",
            TrainingMode.MIDTRAIN: "midtrain",
            TrainingMode.MIDTRAIN_VALIDATION: "midtrain_val",
            TrainingMode.POSTTRAIN: "posttrain",
            TrainingMode.POSTTRAIN_VALIDATION: "posttrain_val",
        }
        return prefixes[self]

    @property
    def display_name(self) -> str:
        names = {
            TrainingMode.PRETRAIN: "Pretraining",
            TrainingMode.PRETRAIN_VALIDATION: "Pretrain Validation",
            TrainingMode.PRETRAIN_EVALUATION: "Pretrain Evaluation",
            TrainingMode.MIDTRAIN: "Midtraining",
            TrainingMode.MIDTRAIN_VALIDATION: "Midtrain Validation",
            TrainingMode.POSTTRAIN: "Post-training",
            TrainingMode.POSTTRAIN_VALIDATION: "Posttrain Validation",
        }
        return names[self]


@dataclass
class EpochResult:
    """Results from running an epoch."""

    mode: TrainingMode
    epoch_num: int
    avg_loss: float
    avg_main_loss: float
    avg_moe_loss: float
    total_steps: int
    total_tokens: int
    elapsed_time: float
    perplexity: float = field(init=False)

    def __post_init__(self):
        self.perplexity = torch.exp(torch.tensor(self.avg_loss)).item()

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.name,
            "epoch": self.epoch_num,
            "avg_loss": self.avg_loss,
            "avg_main_loss": self.avg_main_loss,
            "avg_moe_loss": self.avg_moe_loss,
            "perplexity": self.perplexity,
            "steps": self.total_steps,
            "tokens": self.total_tokens,
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class TrainingResult:
    """Results from a full training run."""

    response: DattaBotAPIResponse
    train_loss: float
    val_loss: float
    total_steps: int
    total_tokens: int
    total_time: float
    interrupted: bool
    vocab: dict


class DattaBotTrainingEngine:
    """
    Unified training engine supporting multiple training modes.
    """

    def __init__(
        self,
        device: torch.device,
        tokenizer: DattaBotTokenizer,
        orig_model: nn.Module,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        components: DattaBotAgentComponents,
        autocast_dtype: torch.dtype,
        metric_tracker=None,
        gpu_profiler=None,
        d_model: int = 1024,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.orig_model = orig_model
        self.model = model
        self.config = get_agent_config()
        self.metric_tracker = metric_tracker
        self.gpu_profiler = gpu_profiler
        self.max_response_tokens = components.max_response_tokens
        self.use_moe = components.use_moe
        self.d_model = d_model
        self.logger = get_logger()
        self.autocast_dtype = autocast_dtype
        self.lr = self.config.agent.lr
        # Training components
        self.loss_fn = components.loss_fn
        self.lr_scheduler = components.lr_scheduler
        self.optimizer = optimizer
        self.scaler = components.scaler
        # Manager for saving checkpoints
        self.chkpt_manager: DattaBotCheckpointManager = get_checkpoint_manager()
        # Mode-specific callbacks
        self._pre_epoch_hooks: dict[TrainingMode, list[Callable]] = {}
        self._post_epoch_hooks: dict[TrainingMode, list[Callable]] = {}
        self._pre_step_hooks: dict[TrainingMode, list[Callable]] = {}
        self._post_step_hooks: dict[TrainingMode, list[Callable]] = {}
        # Training state (reset before each training run)
        self._reset_training_state()

    def _reset_training_state(self) -> None:
        """Reset training state before each training run."""
        self.current_epoch = 0
        self.global_step = 0
        self.tokens_processed = 0
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.best_val_loss = float("inf")

    # =========================================================================
    # Main Training Entry Point
    # =========================================================================

    def train(
        self,
        train_dataloader,
        val_dataloader,
        vocab: dict,
        mode: TrainingMode = TrainingMode.PRETRAIN,
    ) -> TrainingResult:
        """
        Main training loop with early stop conditions.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            vocab: Vocabulary dictionary
            mode: Training mode (PRETRAIN, MIDTRAIN, POSTTRAIN)

        Returns:
            TrainingResult containing all training metrics and response
        """
        self.logger.info(f"Starting {mode.display_name} process.")
        # Reset training state
        self._reset_training_state()
        response = DattaBotAPIResponse()
        response.text = f"Starting {mode.display_name}."
        train_interrupted = False
        # Config values
        max_epochs = self.config.agent.max_training_num_epochs
        max_train_tokens = self.config.agent.max_train_tokens
        max_train_time = self.config.agent.max_train_time_hours * 3600  # to seconds
        num_train_batches = self.config.agent.num_batches_train_every_epoch
        num_val_batches = self.config.agent.num_batches_val_every_epoch
        ckpt_interval = self.config.agent.checkpoint_interval_train_steps
        # Determine validation mode
        val_mode = self._get_validation_mode(mode)
        train_start_time = time.time()

        try:
            self._log_training_start(
                train_dataloader, vocab, num_train_batches, num_val_batches
            )
            for epoch in range(max_epochs):
                self.current_epoch = epoch + 1
                self.logger.debug(f"\nEpoch {self.current_epoch}/{max_epochs}")
                # Update distributed samplers
                self._update_samplers(
                    train_dataloader, val_dataloader, self.current_epoch
                )
                # === Training Phase ===
                if self.gpu_profiler:
                    self.gpu_profiler.log_gpu_memory("Training - before train epoch")
                train_result = self.run_epoch(
                    mode=mode,
                    epoch_num=self.current_epoch,
                    dataloader=train_dataloader,
                    num_batches=num_train_batches,
                )
                if self.gpu_profiler:
                    self.gpu_profiler.log_gpu_memory("Training - after train epoch")
                # Update metrics
                self.global_step += train_result.total_steps
                self.tokens_processed += train_result.total_tokens
                response.num_train_tokens_processed += train_result.total_tokens
                response.num_train_batches += train_result.total_steps
                self.avg_train_loss = train_result.avg_loss
                elapsed = time.time() - train_start_time
                self.logger.debug(
                    f"[Epoch {self.current_epoch}] avg_train_loss={self.avg_train_loss:.4f}, "
                    f"tokens_processed={self.tokens_processed:,}"
                )
                # Check early stop conditions
                should_stop, stop_reason = self._check_stop_conditions(
                    self.tokens_processed, max_train_tokens, elapsed, max_train_time
                )
                if should_stop:
                    self.logger.info(stop_reason)
                    break
                # === Validation Phase ===
                val_result = self.run_epoch(
                    mode=val_mode,
                    epoch_num=self.current_epoch,
                    dataloader=val_dataloader,
                    num_batches=num_val_batches,
                )
                self.tokens_processed += val_result.total_tokens
                response.num_val_tokens_processed += val_result.total_tokens
                response.num_val_batches += val_result.total_steps
                self.avg_val_loss = val_result.avg_loss
                self.logger.debug(
                    f"[Epoch {self.current_epoch}] avg_val_loss={self.avg_val_loss:.4f}"
                )
                # Log epoch metrics
                self._log_epoch_metrics(train_result, val_result, self.tokens_processed)
                # Save best model
                if self.avg_val_loss < self.best_val_loss:
                    self.best_val_loss = self.avg_val_loss
                    self.logger.info(
                        f"New best model! val_loss={self.avg_val_loss:.4f}"
                    )
                    self._save_checkpoint(checkpoint_type="best")
                # Periodic checkpoint
                if self.global_step % ckpt_interval == 0:
                    self._save_checkpoint(checkpoint_type="periodic")
            # Training complete
            response.text = self._build_success_message(response)
        except KeyboardInterrupt:
            train_interrupted = True
            response.text = "Training interrupted by user (ctrl+c)."
            self.logger.error(response.text)
        except Exception as e:
            train_interrupted = True
            response.text = f"Training stopped unexpectedly: {e}"
            self.logger.error(response.text)
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
        finally:
            # Always save final checkpoint
            self._save_checkpoint(checkpoint_type="final")

        total_time = time.time() - train_start_time
        return TrainingResult(
            response=response,
            train_loss=self.avg_train_loss,
            val_loss=self.avg_val_loss,
            total_steps=self.global_step,
            total_tokens=self.tokens_processed,
            total_time=total_time,
            interrupted=train_interrupted,
            vocab=vocab,
        )

    # =========================================================================
    # Training Helpers
    # =========================================================================

    def _get_validation_mode(self, train_mode: TrainingMode) -> TrainingMode:
        """Get the corresponding validation mode for a training mode."""
        mode_map = {
            TrainingMode.PRETRAIN: TrainingMode.PRETRAIN_VALIDATION,
            TrainingMode.MIDTRAIN: TrainingMode.MIDTRAIN_VALIDATION,
            TrainingMode.POSTTRAIN: TrainingMode.POSTTRAIN_VALIDATION,
        }
        return mode_map.get(train_mode, TrainingMode.PRETRAIN_VALIDATION)

    def _update_samplers(self, train_loader, val_loader, epoch: int) -> None:
        """Update distributed samplers for the new epoch."""
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

    def _check_stop_conditions(
        self,
        tokens_processed: int,
        max_tokens: int,
        elapsed: float,
        max_time: float,
    ) -> tuple[bool, str]:
        """Check if training should stop early."""
        if max_tokens != -1 and tokens_processed >= max_tokens:
            return True, f"Reached {tokens_processed:,} tokens. Stopping training."
        if max_time != -1 and elapsed >= max_time:
            return True, f"Reached {elapsed/3600:.2f} hours. Stopping training."
        return False, ""

    def _log_training_start(
        self,
        train_loader,
        vocab: dict,
        num_train_batches: int,
        num_val_batches: int,
    ) -> None:
        """Log training start information."""
        self.logger.info(
            f"Got training and validation data for dataset {train_loader.dataset_type}. "
            f"Now going into training loop for {self.config.agent.max_training_num_epochs} epochs, "
            f"{num_train_batches} batches per training epoch, "
            f"and {num_val_batches} batches per validation epoch. "
            f"Length of vocab: {len(vocab)}. "
            f"Batch size: {self.config.agent.batch_size}."
        )

    def _log_epoch_metrics(
        self,
        train_result: EpochResult,
        val_result: EpochResult,
        tokens_processed: int,
    ) -> None:
        """Log epoch metrics to metric tracker."""
        if not is_rank_0() or not self.metric_tracker:
            return

        self.metric_tracker.log_metrics(
            {
                f"{train_result.mode.metric_prefix}/loss": train_result.avg_loss,
                f"{val_result.mode.metric_prefix}/loss": val_result.avg_loss,
                f"{train_result.mode.metric_prefix}/tokens_processed": tokens_processed,
                f"{train_result.mode.metric_prefix}/perplexity": train_result.perplexity,
                f"{val_result.mode.metric_prefix}/perplexity": val_result.perplexity,
            },
            step=None,
        )

    def _save_checkpoint(self, checkpoint_type: str) -> None:
        """Save a checkpoint for the model."""
        # Update bundle state from training engine state
        self.chkpt_manager.bundle.update_state(
            epoch=self.current_epoch,
            global_step=self.global_step,
            tokens_processed=self.tokens_processed,
            train_loss=self.avg_train_loss,
            val_loss=self.avg_val_loss,
        )
        self.chkpt_manager.save_agent()
        self.logger.info(f"Saved model ({checkpoint_type}) at step {self.global_step}")

    def _build_success_message(self, response: DattaBotAPIResponse) -> str:
        """Build success message for completed training."""
        base_msg = (
            f"Successfully trained on {response.num_train_batches} batches. "
            f"Validated on {response.num_val_batches} batches."
        )
        if self.metric_tracker and self.metric_tracker.active:
            return f"{base_msg} View training progress at: {self.metric_tracker.get_run_url()}"
        return base_msg

    # =========================================================================
    # Epoch Running (existing code)
    # =========================================================================

    def run_epoch(
        self,
        mode: TrainingMode,
        epoch_num: int,
        dataloader,
        num_batches: int,
    ) -> EpochResult:
        """Run a single epoch in the specified mode."""
        start_time = time.time()
        is_training = mode.is_training
        self._run_hooks("pre_epoch", mode, epoch_num=epoch_num)
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        # Initialize accumulators
        accumulated_loss = torch.tensor(0.0, device=self.device)
        accumulated_main_loss = torch.tensor(0.0, device=self.device)
        accumulated_moe_loss = torch.tensor(0.0, device=self.device)
        accumulated_tokens = torch.tensor(0, device=self.device)
        total_loss = 0.0
        total_main_loss = 0.0
        total_moe_loss = 0.0
        total_steps = 0
        total_tokens = 0
        sync_interval = self.config.agent.training_sync_interval
        progress_bar = tqdm.tqdm(
            enumerate(dataloader),
            total=num_batches,
            desc=f"{mode.display_name} epoch {epoch_num}",
            leave=True,
            disable=not is_rank_0(),
        )
        context = torch.no_grad() if not is_training else nullcontext()

        with context:
            for batch_idx, batch in progress_bar:
                if batch_idx >= num_batches:
                    break
                self._run_hooks("pre_step", mode, epoch_num=epoch_num, step=batch_idx)
                # Prepare batch
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                batch_size = input_ids.shape[0]
                attention_pad_mask = self._get_attention_pad_mask(input_ids)
                if is_training:
                    self.optimizer.zero_grad()
                # Forward pass
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=is_autocast_enabled(self.device),
                    dtype=self.autocast_dtype,
                ):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_pad_mask=attention_pad_mask,
                    )
                    padding_mask = labels != self.tokenizer.pad_token_id
                    main_loss, moe_loss = self._compute_loss(
                        outputs=logits,
                        targets=labels,
                        mask=padding_mask,
                    )
                    if moe_loss is not None:
                        loss = (
                            main_loss
                            + self.config.neural_net.moe_load_balance_weight * moe_loss
                        )
                    else:
                        loss = main_loss
                # Backward pass (training only)
                if is_training:
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.config.agent.max_grad_norm:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.agent.max_grad_norm,
                        )
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif HAS_XLA:
                        xm.optimizer_step(self.optimizer)
                        xm.mark_step()
                    else:
                        self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                else:
                    if HAS_XLA:
                        xm.mark_step()

                # Accumulate metrics
                total_steps += 1
                accumulated_loss += loss.detach()
                accumulated_main_loss += main_loss.detach()
                if moe_loss is not None:
                    accumulated_moe_loss += moe_loss.detach()
                accumulated_tokens += padding_mask.sum()
                # Sync periodically
                if total_steps % sync_interval == 0:
                    total_loss, total_main_loss, total_moe_loss, total_tokens = (
                        self._sync_metrics(
                            accumulated_loss,
                            accumulated_main_loss,
                            accumulated_moe_loss,
                            accumulated_tokens,
                            total_steps,
                        )
                    )
                    avg_loss = total_loss / total_steps
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ppl": f"{perplexity:.2f}",
                        }
                    )
                self._run_hooks(
                    "post_step", mode, epoch_num=epoch_num, step=batch_idx, loss=loss
                )

        # Final sync
        total_loss, total_main_loss, total_moe_loss, total_tokens = self._sync_metrics(
            accumulated_loss,
            accumulated_main_loss,
            accumulated_moe_loss,
            accumulated_tokens,
            total_steps,
        )
        avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
        avg_main_loss = (
            total_main_loss / total_steps if total_steps > 0 else float("inf")
        )
        avg_moe_loss = total_moe_loss / total_steps if total_steps > 0 else 0.0
        # Distributed sync
        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(
                [total_loss, float(total_steps), float(total_tokens)],
                device=self.device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, total_steps, total_tokens = loss_tensor.tolist()
            total_steps = int(total_steps)
            total_tokens = int(total_tokens)
            avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
        elapsed_time = time.time() - start_time
        self._run_hooks("post_epoch", mode, epoch_num=epoch_num)
        return EpochResult(
            mode=mode,
            epoch_num=epoch_num,
            avg_loss=avg_loss,
            avg_main_loss=avg_main_loss,
            avg_moe_loss=avg_moe_loss,
            total_steps=total_steps,
            total_tokens=total_tokens,
            elapsed_time=elapsed_time,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def register_hook(
        self, hook_type: str, mode: TrainingMode, callback: Callable
    ) -> None:
        hooks_map = {
            "pre_epoch": self._pre_epoch_hooks,
            "post_epoch": self._post_epoch_hooks,
            "pre_step": self._pre_step_hooks,
            "post_step": self._post_step_hooks,
        }
        if hook_type not in hooks_map:
            raise ValueError(f"Unknown hook type: {hook_type}")
        if mode not in hooks_map[hook_type]:
            hooks_map[hook_type][mode] = []
        hooks_map[hook_type][mode].append(callback)

    def _run_hooks(self, hook_type: str, mode: TrainingMode, **kwargs) -> None:
        hooks_map = {
            "pre_epoch": self._pre_epoch_hooks,
            "post_epoch": self._post_epoch_hooks,
            "pre_step": self._pre_step_hooks,
            "post_step": self._post_step_hooks,
        }
        hooks = hooks_map.get(hook_type, {}).get(mode, [])
        for hook in hooks:
            hook(**kwargs)

    def _get_attention_pad_mask(self, input_ids: Tensor) -> Tensor:
        return input_ids == self.tokenizer.pad_token_id

    def _compute_loss(
        self, outputs: Tensor, targets: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Compute loss using loss function.
        Args:
            outputs: Tensor of shape (batch_size, seq_len, d_model)
            targets: Tensor of shape (batch_size, seq_len)
            mask: Tensor of shape (batch_size, seq_len)
        Returns:
            main_loss: scalar tensor (cross-entropy loss)
            moe_loss: scalar tensor or None (load balancing loss)
        """
        # Flatten to (batch*seq_len, d_model).
        batch_size, seq_len, d_model = outputs.size()
        outputs_flat = outputs.view(batch_size * seq_len, d_model)
        targets_flat = targets.view(batch_size * seq_len)
        if mask is not None:
            mask_flat = mask.view(batch_size * seq_len)
            outputs_flat = outputs_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        outputs_flat = outputs_flat.to(self.device).float()
        targets_flat = targets_flat.to(self.device)
        # Main cross-entropy loss.
        if outputs_flat.numel() == 0:
            main_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            main_loss = self.loss_fn(outputs_flat, targets_flat).loss
        # MoE auxiliary loss (only during training).
        moe_loss = None
        if self.use_moe:
            if isinstance(
                self.model,
                (
                    torch.nn.DataParallel,
                    torch.nn.parallel.DistributedDataParallel,
                ),
            ):
                moe_loss = self.model.module.get_load_balancing_loss()
            else:
                moe_loss = self.model.get_load_balancing_loss()
        return main_loss, moe_loss

    def _sync_metrics(
        self,
        accumulated_loss: Tensor,
        accumulated_main_loss: Tensor,
        accumulated_moe_loss: Tensor,
        accumulated_tokens: Tensor,
        total_steps: int,
    ) -> tuple[float, float, float, int]:
        return (
            accumulated_loss.item(),
            accumulated_main_loss.item(),
            accumulated_moe_loss.item(),
            int(accumulated_tokens.item()),
        )

    def _profile_training(self, num_steps=20, log_dir: Optional[str] = None):
        """
        Create an AgentProfiler and run the training profiler.
        """
        from src.agent_profiler import AgentProfiler

        profiler = AgentProfiler(self, log_dir=log_dir)
        return profiler.profile_training(num_steps=num_steps)
