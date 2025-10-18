import time
import traceback
from math import ceil
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from src.agent_config import get_agent_config
from src.api_interface import DattaBotAPIResponse
from src.checkpointing import load_agent, save_agent
from src.communication_mgr import DattaBotCommunicationManager
from src.data_loader import DattabotDataBuilder

from src.gpu_profiler import BackgroundGPUProfiler
from src.logger import get_logger
from src.metric_tracker import get_metric_tracker, MetricTracker
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer
from src.util import (
    get_logging_level_from_config,
    get_tensor_dtype_from_config,
    is_device_cpu,
    is_rank_0,
    setup_torch_dist_init,
)

from torch import nn, Tensor
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
from torch.utils.data.distributed import DistributedSampler


class Agent:
    def __init__(self) -> None:
        # Initialize the default distributed process group before doing anything else.
        setup_torch_dist_init()
        self.local_rank = 0
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.metric_tracker: MetricTracker | None = None
        self.gpu_profiler = None
        # For file names, start with this prefix
        self.agent_name = self.config.agent.agent_name
        self.data_dir = self.config.agent.data_directory
        self.plot_dir = self.config.agent.plot_directory
        self.tensor_dtype = get_tensor_dtype_from_config(self.config)
        # Setup hardware settings.
        self.agent_device = self.config.env.device
        self._setup_cpu_gpu_settings()
        # Setup multi-gpu settings.
        if dist.is_available() and torch.cuda.device_count() > 1:
            self._setup_distributed()
        # Setup tokenizer.
        self.tokenizer = get_tokenizer(encoding_name="o200k_harmony")
        self.comm_manager = DattaBotCommunicationManager()
        tokenizer_model_name = repr(self.tokenizer)
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        self.d_model = self.config.neural_net.model_dimensions
        # Setup data loader.
        self.data_builder = DattabotDataBuilder()
        # Initialize AMP scaler.
        # Reduce memory consumption and improve training speed.
        # https://arxiv.org/abs/1710.03740
        self.scaler = torch.amp.GradScaler(device=self.agent_device)
        # Initialize AdaptiveLogSoftmaxWithLoss.
        # Vocab size is huge and increases the memory during forward pass by a
        # lot, to reduce the memory overhead we use AdaptiveLogSoftmaxWithLoss
        # rather than standard softmax. Tradeoff between accuracy and memory.
        # https://arxiv.org/abs/1609.04309
        self.loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.d_model,
            n_classes=self.tokenizer.vocab_size,
            cutoffs=[10_000, 50_000, self.tokenizer.vocab_size - 1],
            div_value=4.0,
            head_bias=False,
        ).to(self.agent_device)
        # Setup training objects.
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.batch_size = self.config.agent.batch_size
        self.max_response_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Batch size: {self.batch_size}")
        self.logger.debug(f"Max tokens: {self.max_response_tokens}")
        self.lr = self.config.agent.lr
        # Setup model.
        # We pass in tensor_dtype to the model, but remember we may use
        # AMP autocast during model inference/training which will have
        # all tensors as float16/bfloat16.
        self.model = DattaBotModel(device=self.agent_device, dtype=self.tensor_dtype)
        self.model.to(device=self.agent_device, dtype=self.tensor_dtype)
        self.logger.info(f"Model is on: {self.agent_device}")
        self.logger.info(f"Model dimensions: {self.d_model}")
        self.logger.info(
            f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        # TODO(PiyushDatta): Try and get Shampoo optimizer to work.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.config.agent.weight_decay,
        )
        # Only training epochs, do not include validation epochs.
        self.lr_scheduler = TorchOneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.config.agent.max_training_num_epochs,
            steps_per_epoch=self.config.agent.num_batches_train_every_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
        )
        # Load model weights
        self.weights_fname = self.config.agent.weights_file_name
        load_agent(
            model=self.model,
            filepath=self.weights_fname,
            device=self.agent_device,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
        )
        # Wrap with distributed layer if we have multiple gpus.
        if dist.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        self.model.cuda()
        # Setup inference engine when needed.
        self.inference_engine = None
        self.use_moe = self.config.neural_net.use_moe
        self.moe_weight = self.config.neural_net.moe_load_balance_weight
        self.training = False

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up distributed process group if initialized."""
        if dist.is_available() and dist.is_initialized():
            self.logger.info("Destroying distributed process group...")
            dist.destroy_process_group()

    def _save_agent(self):
        """Helper method to save agent."""
        save_agent(
            model=self.model,
            filename=self.weights_fname,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
        )

    def _setup_inference_engine(self):
        """Setup high-performance inference engine."""
        from src.inference_engine import DattaBotInferenceEngine

        if self.inference_engine is None:
            self.inference_engine = DattaBotInferenceEngine(
                model=self.model,
                device=self.agent_device,
                adaptive_softmax=self.loss_fn,
            )
            self.logger.info("Inference engine initialized and ready for deployment!")

    def _setup_distributed(self):
        assert (
            dist.is_initialized() == True
        ), "Distributed process group is not initialized. Please call dist.is_initialized() first."
        # Ensure this is running under torchrun.
        self.local_rank = dist.get_rank()
        self.logger.debug(
            f"Setting up setup_distributed with local_rank={self.local_rank}"
        )
        torch.cuda.set_device(self.local_rank)
        self.agent_device = f"cuda:{self.local_rank}"

    def _setup_cpu_gpu_settings(self):
        # torch.manual seed(3407) is all you need
        # https://arxiv.org/pdf/2109.08203
        seed = 3407
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.agent_device = self.config.env.device
        torch.cuda.empty_cache()

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
        outputs_flat = outputs_flat.to(self.agent_device)
        targets_flat = targets_flat.to(self.agent_device)
        # Main cross-entropy loss
        loss_output = self.loss_fn(outputs_flat, targets_flat)
        main_loss = loss_output.loss
        # MoE auxiliary loss (only during training)
        moe_loss = None
        if self.training and self.use_moe:
            if isinstance(
                self.model,
                (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
            ):
                moe_loss = self.model.module.get_load_balancing_loss()
            else:
                moe_loss = self.model.get_load_balancing_loss()
        return main_loss, moe_loss

    def _get_attention_pad_mask(self, input_ids: Tensor) -> Tensor:
        """
        Get attention pad mask.
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            mask: Tensor of shape (batch, 1, seq_len, seq_len)
        """
        # TODO(PiyushDatta): Figure out what to do with the mask, we are padding the sequences.
        #                    This is a temporary solution to unblock the flash attention training.
        return None
        # Attention mask (1 for real tokens, 0 for pad)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # [batch, seq_len]
        attention_pad_mask = (input_ids != self.tokenizer.pad_token_id).to(
            self.agent_device
        )
        # unsqueeze to create head dim
        # [batch, seq_len] -> [batch, 1, seq_len]
        attention_pad_mask = attention_pad_mask.unsqueeze(1)
        # unsqueeze again to create key/value dim
        # [batch, 1, seq_len] -> [batch, 1, seq_len, 1]
        attention_pad_mask = attention_pad_mask.unsqueeze(-1)
        # expand the last dim to seq_len
        # [batch, 1, seq_len, 1] -> [batch, 1, seq_len, seq_len]
        attention_pad_mask = attention_pad_mask.expand(-1, -1, -1, seq_len)
        expected_shape = (batch_size, 1, seq_len, seq_len)
        assert attention_pad_mask.shape == expected_shape, (
            f"Attention mask shape mismatch! "
            f"Got {attention_pad_mask.shape}, expected {expected_shape}"
        )
        return attention_pad_mask

    def _get_gpu_info(self) -> list[str]:
        gpu_name = "Could not retrieve gpu_name"
        total_memory = "Could not retrieve total_memory"
        total_cores = "Could not retrieve total_cores"
        if torch.cuda.is_available() and self.agent_device != "cpu":
            gpu_name = torch.cuda.get_device_name(self.agent_device)
            # Convert bytes to MB
            total_memory = torch.cuda.get_device_properties(
                self.agent_device
            ).total_memory // (1024**2)
            total_cores = torch.cuda.get_device_properties(
                self.agent_device
            ).multi_processor_count
        return gpu_name, total_memory, total_cores

    @property
    def tokenizer_obj(self):
        """
        Accessor for the tokenizer object.
        Example usage: agent.tokenizer_obj.encode(["Hello"])
        """
        return self.tokenizer

    def new_training_session(self):
        self.logger.info("Agent's training session has begun...")
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        # Setup MetricTracker (w&b, tensorboard, or plain csv).
        self.metric_tracker = get_metric_tracker(
            project=f"{self.config.env.env_name}",
            run_name=f"{self.config.env.env_name}_training_run_{int(time.time())}",
        )
        if self.metric_tracker.active:
            self.metric_tracker.log_config(
                {
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "epochs": self.config.agent.max_training_num_epochs,
                    "model_dims": self.config.neural_net.model_dimensions,
                    "dataset": self.config.env.dataset_name,
                }
            )
            self.logger.info(
                f"Training session can be monitored here: {self.metric_tracker.get_run_url()}"
            )
        # GPU profiler.
        self.gpu_profiler = BackgroundGPUProfiler(
            device=self.agent_device,
            sample_every_x_seconds=1.0,
        )
        self.gpu_profiler.start()
        mp.set_start_method("spawn", force=True)
        torch.cuda.reset_peak_memory_stats(self.agent_device)
        self.training = True

    def end_training_session(self):
        self.logger.info("Agent's training session has ended.")
        if self.metric_tracker and self.metric_tracker.active:
            self.logger.info(
                f"Training session can be monitored here: {self.metric_tracker.get_run_url()}"
            )
        if self.gpu_profiler:
            self.gpu_profiler.stop()
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.training = False

    def train_agent(self) -> DattaBotAPIResponse:
        """
        Main training loop with early stop conditions:
        - max_train_tokens
        - max_train_time_hours
        Includes:
        - periodic checkpoint saving
        - best model saving
        - final model saving
        """
        self.logger.info("Starting training process.")
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        response.text = "Starting training."
        train_interrupted = False
        avg_train_loss = 0
        avg_val_loss = 0
        vocab: dict[str, int] = {}
        try:
            self.new_training_session()
            self.gpu_profiler.log_gpu_memory("Training - start training")
            tokens_processed = 0
            tokens_per_batch = self.batch_size * self.config.agent.max_response_tokens
            max_train_tokens = self.config.agent.max_train_tokens
            max_train_time = self.config.agent.max_train_time_hours
            total_steps = 0
            ckpt_interval = self.config.agent.checkpoint_interval_train_steps
            # seconds
            max_train_time = max_train_time * 3600
            # Record start time
            train_start_time = time.time()
            # Set an arbitrary starting best val loss amount.
            best_val_loss: float = float("inf")
            num_train_batches_per_phase: int = (
                self.config.agent.num_batches_train_every_epoch
            )
            num_val_batches_per_phase: int = (
                self.config.agent.num_batches_val_every_epoch
            )
            # Set up data loaders
            self.gpu_profiler.log_gpu_memory("Training - before setup data")
            self.logger.info("Setting up data.")
            train_dataloader, val_dataloader, vocab = self.data_builder.setup_data()
            self.gpu_profiler.log_gpu_memory("Training - after setup data")
            # Actual training algorithm.
            self.logger.info(
                f"Got training and validation data for dataset {train_dataloader.dataset_type}. Now going into training loop for "
                f"{self.config.agent.max_training_num_epochs} epochs, "
                f"{num_train_batches_per_phase} batches per training epoch, "
                f"and {num_val_batches_per_phase} batches per validation epoch. "
                f"Length of vocab: {len(vocab)}. "
                f"Batch size: {self.batch_size}."
            )
            for epoch in range(self.config.agent.max_training_num_epochs):
                curr_epoch_num = epoch + 1
                if is_rank_0():
                    self.logger.debug(
                        f"\nEpoch {curr_epoch_num}/{self.config.agent.max_training_num_epochs}"
                    )
                # Tell DistributedSampler to shuffle differently this epoch.
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(curr_epoch_num)
                if isinstance(val_dataloader.sampler, DistributedSampler):
                    val_dataloader.sampler.set_epoch(curr_epoch_num)
                self.gpu_profiler.log_gpu_memory("Training - before train epoch")
                # === Training ===
                avg_train_loss, train_steps_this_epoch, train_tokens_in_epoch = (
                    self._train_epoch(
                        epoch_num=curr_epoch_num,
                        dataloader=train_dataloader,
                        num_batches=num_train_batches_per_phase,
                    )
                )
                self.gpu_profiler.log_gpu_memory("Training - after train epoch")
                total_steps += train_steps_this_epoch
                # Update tokens processed
                tokens_processed += train_tokens_in_epoch
                response.num_train_tokens_processed += train_tokens_in_epoch
                response.num_train_batches += train_steps_this_epoch
                elapsed = time.time() - train_start_time
                if is_rank_0():
                    self.logger.debug(
                        f"[Epoch {curr_epoch_num}] avg_train_loss={avg_train_loss:.4f}, "
                        f"tokens_processed={tokens_processed:,}"
                    )
                # Stop conditions
                if max_train_tokens != -1 and tokens_processed >= max_train_tokens:
                    self.logger.info(
                        f"Reached {tokens_processed:,} tokens. Stopping training."
                    )
                    break
                if max_train_time != -1 and elapsed >= max_train_time:
                    self.logger.info(
                        f"Reached {elapsed/3600:.2f} hours. Stopping training."
                    )
                    break
                # === Validation ===
                avg_val_loss, val_steps_this_epoch, val_tokens_in_epoch = (
                    self._val_epoch(
                        epoch_num=curr_epoch_num,
                        dataloader=val_dataloader,
                        num_batches=num_val_batches_per_phase,
                    )
                )
                tokens_processed += val_tokens_in_epoch
                response.num_val_tokens_processed += val_tokens_in_epoch
                response.num_val_batches += val_steps_this_epoch
                if is_rank_0():
                    self.logger.debug(
                        f"[Epoch {curr_epoch_num}] avg_val_loss={avg_val_loss:.4f}"
                    )
                    # Log epoch metrics to MetricTracker
                    self.metric_tracker.log_metrics(
                        {
                            "train/loss": avg_train_loss,
                            "val/loss": avg_val_loss,
                            "train/tokens_processed": tokens_processed,
                        },
                        step=None,
                    )
                # Save best model.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if is_rank_0():
                        self._save_agent()
                        self.logger.info("New best model saved!")
                    self.logger.info(f"Best validation loss: {avg_val_loss:.2f}\n")
                # Periodic checkpoint.
                if is_rank_0() and total_steps % ckpt_interval == 0:
                    if is_rank_0():
                        self._save_agent()
                        self.logger.info(
                            f"New model saved since total_steps ({total_steps}) has hit ckpt_interval ({ckpt_interval})."
                        )
            # Done training!
            if self.metric_tracker.active:
                response.text = (
                    f"Successfully trained on {response.num_train_batches} batches. "
                    f"Validated on {response.num_val_batches} batches. "
                    f"View training progress at: {self.metric_tracker.get_run_url()}"
                )
            else:
                response.text = (
                    f"Successfully trained on {response.num_train_batches} batches. "
                    f"Validated on {response.num_val_batches} batches."
                )
        except KeyboardInterrupt:
            # Stopped training because we cancelled it via ctrl+c (KeyboardInterrupt).
            train_interrupted = True
            err_msg = f"Training interrupted by user (ctrl+c)."
            response.text = err_msg
            self.logger.error(err_msg)
            self.end_training_session()
        except Exception as e:
            err_msg = f"Stopped training! Something unexpected happened: {e}"
            tb_str = traceback.format_exc()
            self.logger.error(err_msg)
            self.logger.error(f"An error occurred during training:\n{tb_str}")
            response.text = err_msg
            self.end_training_session()
        finally:
            # Always save the final model.
            if is_rank_0():
                self._save_agent()
        # Log the training details, including the time taken to train.
        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time
        # Calculate the total number of model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_millions = total_params // 1_000_000
        self._log_training_details(
            num_train_tokens_processed=response.num_train_tokens_processed,
            num_val_tokens_processed=response.num_val_tokens_processed,
            num_train_batches=response.num_train_batches,
            num_val_batches=response.num_val_batches,
            training_score=avg_train_loss,
            validation_score=avg_val_loss,
            dataset_name=train_dataloader.dataset_type.value,
            model_name=self.agent_name,
            vocab=vocab,
            total_training_time=round(total_training_time, 2),
            total_params_millions=total_params_millions,
            interrupted=train_interrupted,
        )
        self.end_training_session()
        return response

    def _train_epoch(
        self, epoch_num: int, dataloader, num_batches: int
    ) -> tuple[float, int, int]:
        """
        Run one training epoch.

        Returns:
            avg_loss: Average loss for this epoch
            steps: Number of batches processed
            tokens_processed: Number of tokens processed
        """
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_moe_loss = 0.0
        total_steps = 0
        total_tokens = 0
        log_and_plot_every_x_steps = (
            self.config.agent.logging_and_plotting_every_x_steps
        )
        # Setup progress bar
        progress_bar = tqdm.tqdm(
            range(num_batches), desc=f"Training for epoch {epoch_num}", leave=True
        )
        # Go through all the batches.
        for _ in progress_bar:
            batch = next(dataloader)
            # input_ids: [batch, seq_len]
            input_ids = batch[0].to(self.agent_device)
            labels = batch[1].to(self.agent_device)
            batch_size = input_ids.shape[0]
            attention_pad_mask = self._get_attention_pad_mask(input_ids=input_ids)
            # Zero the gradients.
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.agent_device,
                enabled=(not is_device_cpu(self.agent_device)),
                dtype=torch.bfloat16,
            ):
                # Forward pass.
                logits = self.model(
                    input_ids=input_ids, attention_pad_mask=attention_pad_mask
                )
                assert logits.shape == (
                    batch_size,
                    self.max_response_tokens,
                    self.d_model,
                ), f"Model output/Logits shape mismatch. Logits shape: {logits.shape}, expected: ({batch_size}, {self.max_response_tokens}, {self.d_model})"
                self.logger.debug(
                    f"logits.shape = {logits.shape}, labels.shape = {labels.shape}"
                )
                # Calculate loss.
                self.gpu_profiler.log_gpu_memory("Training - before loss")
                padding_mask = labels != self.tokenizer.pad_token_id
                main_loss, moe_loss = self._compute_loss(
                    outputs=logits, targets=labels, mask=padding_mask
                )
                if moe_loss is not None:
                    loss = main_loss + self.moe_weight * moe_loss
                else:
                    loss = main_loss
                self.gpu_profiler.log_gpu_memory("Training - after loss")
            # Backprop.
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Clip gradients to prevent exploding gradients.
            if self.config.agent.max_grad_norm:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.agent.max_grad_norm
                )
            # Update weights.
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            # Update learning rate if using a scheduler.
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # Update metrics
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            if moe_loss is not None:
                total_moe_loss += moe_loss.item()
            total_steps += 1
            # Count only non-pad tokens
            total_tokens += padding_mask.sum().item()
            avg_loss = total_loss / total_steps
            avg_main_loss = total_main_loss / total_steps
            avg_moe_loss = total_moe_loss / total_steps if moe_loss is not None else 0.0
            loss_per_token = (
                total_loss / total_tokens if total_tokens > 0 else float("inf")
            )
            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": avg_loss,
                    "main_loss": avg_main_loss,
                    "moe_loss": avg_moe_loss,
                    "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
                }
            )
            # Log metrics using MetricTracker
            if is_rank_0():
                metrics = {
                    "train/batch_loss": loss.item(),
                    "train/batch_main_loss": main_loss.item(),
                    "train/batch_perplexity": torch.exp(main_loss).item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/loss_per_token": loss_per_token,
                }
                if moe_loss is not None:
                    metrics["train/batch_moe_loss"] = moe_loss.item()
                    metrics["train/moe_weight"] = self.moe_weight
                self.metric_tracker.log_metrics(metrics, step=None)
                if total_steps % log_and_plot_every_x_steps == 0:
                    moe_str = (
                        f", moe_loss={moe_loss.item():.4f}"
                        if moe_loss is not None
                        else ""
                    )
                    self.logger.info(
                        f"[Epoch {epoch_num} | Step {total_steps}] "
                        f"total_loss={loss.item():.4f}, "
                        f"main_loss={main_loss.item():.4f}{moe_str}, "
                        f"perplexity={torch.exp(main_loss).item():.2f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                        f"loss/token={loss_per_token:.4f}"
                    )
                    latest_metrics = self.gpu_profiler.get_latest_metrics()
                    if latest_metrics:
                        self.logger.info(
                            f"Epoch {epoch_num} GPU Metrics - "
                            f"Memory Allocated: {latest_metrics.memory_allocated} MB, "
                            f"Utilization: {latest_metrics.utilization}%, "
                            f"CPU Usage: {latest_metrics.cpu_percent}%, "
                            f"RAM Usage: {latest_metrics.ram_percent}%"
                        )
        # Sync loss across all ranks.
        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(
                [total_loss, total_steps, total_tokens], device=self.agent_device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, total_steps, total_tokens = loss_tensor.tolist()
            avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
        return avg_loss, total_steps, total_tokens

    def _val_epoch(
        self, epoch_num: int, dataloader, num_batches: int
    ) -> tuple[float, int, int]:
        """
        Run one validation epoch.

        Returns:
            avg_loss: Average validation loss
            steps: Number of batches processed
            tokens_processed: Number of tokens processed
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        total_tokens = 0
        # Setup progress bar
        progress_bar = tqdm.tqdm(
            range(num_batches), desc=f"Validating for epoch {epoch_num}", leave=True
        )
        # Go through all the batches.
        # Disable gradient calculations and go through all the batches.
        with torch.no_grad():
            for _ in progress_bar:
                batch = next(dataloader)
                input_ids, labels = batch[0].to(self.agent_device), batch[1].to(
                    self.agent_device
                )
                batch_size = input_ids.shape[0]
                attention_pad_mask = self._get_attention_pad_mask(input_ids=input_ids)
                with torch.autocast(
                    device_type=self.agent_device,
                    enabled=(not is_device_cpu(self.agent_device)),
                    dtype=torch.bfloat16,
                ):
                    # Forward pass.
                    logits = self.model(
                        input_ids=input_ids, attention_pad_mask=attention_pad_mask
                    )
                    assert logits.shape == (
                        batch_size,
                        self.max_response_tokens,
                        self.d_model,
                    ), f"Model output/Logits shape mismatch. Logits shape: {logits.shape}, expected: ({batch_size}, {self.max_response_tokens}, {self.d_model})"
                    # Calculate loss.
                    padding_mask = labels != self.tokenizer.pad_token_id
                    loss, _ = self._compute_loss(
                        outputs=logits, targets=labels, mask=padding_mask
                    )
                total_loss += loss.item()
                total_steps += 1
                # Count only non-pad tokens
                total_tokens += padding_mask.sum().item()
                avg_loss = total_loss / total_steps
                loss_per_token = (
                    total_loss / total_tokens if total_tokens > 0 else float("inf")
                )
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                # Update progress bar.
                progress_bar.set_postfix(
                    {
                        "val_loss": avg_loss,
                        "val_perplexity": perplexity,
                    }
                )
                if is_rank_0():
                    # Log validation batch metrics.
                    self.metric_tracker.log_metrics(
                        {
                            "val/batch_loss": loss.item(),
                            "val/batch_perplexity": torch.exp(loss).item(),
                            "val/loss_per_token": loss_per_token,
                        },
                        step=None,
                    )
        # Sync loss across all ranks.
        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(
                [total_loss, total_steps, total_tokens], device=self.agent_device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, total_steps, total_tokens = loss_tensor.tolist()
            avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
        # Log epoch summary if rank 0.
        if is_rank_0():
            self.logger.info(
                f"[Validation | Epoch {epoch_num}] avg_loss={avg_loss:.4f}, "
                f"perplexity={torch.exp(torch.tensor(avg_loss)).item():.2f}"
            )
            self.metric_tracker.log_metrics(
                {
                    "val/loss": avg_loss,
                    "val/perplexity": perplexity,
                },
                step=None,
            )

        return avg_loss, total_steps, total_tokens

    def respond_to_queries(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        if is_rank_0():
            self.logger.info(f"Processing queries: {queries}")
            self.logger.info(f"Queries length: {len(queries)}")
        # Call the inference engine.
        if self.inference_engine is None:
            self._setup_inference_engine()
        assert self.inference_engine is not None, "Inference engine is not set up."
        # Use inference engine for batch generation.
        responses: list[DattaBotAPIResponse] = self.inference_engine.batch_generate(
            queries=queries
        )
        # Validate responses.
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert all(
            isinstance(r, DattaBotAPIResponse) for r in responses
        ), "All responses must be DattaBotAPIResponse"
        # Log the results. Only on rank 0.
        if is_rank_0():
            self.logger.debug(f"Query Response for first response: {responses[0].text}")
            self.logger.debug(
                f"Number of Batches for first response: {responses[0].num_train_batches}"
            )
            self.logger.debug(
                f"Tensor Response for the first response: {responses[0].tensor_response}"
            )
            self.logger.debug(f"Number of responses: {len(responses)}")
        return responses

    def _log_training_details(
        self,
        num_train_tokens_processed: int,
        num_val_tokens_processed: int,
        num_train_batches: int,
        num_val_batches: int,
        training_score: float,
        validation_score: float,
        dataset_name: str,
        model_name: str,
        vocab: dict[str, int],
        total_training_time: float,
        total_params_millions: int,
        interrupted=False,
    ):
        """Log final training summary using MetricTracker (W&B or other backend)."""
        gpu_name, total_memory, total_cores = self._get_gpu_info()

        if self.metric_tracker.active:
            self.metric_tracker.log_metrics(
                {
                    "summary/model_name": str(model_name),
                    "summary/dataset_name": str(dataset_name),
                    "summary/vocab_length": len(vocab),
                    "summary/batch_size": self.batch_size,
                    "summary/train_batches_completed": num_train_batches,
                    "summary/val_batches_completed": num_val_batches,
                    "summary/train_tokens_processed": num_train_tokens_processed,
                    "summary/val_tokens_processed": num_val_tokens_processed,
                    "summary/train_loss": training_score,
                    "summary/val_loss": validation_score,
                    "summary/total_params_millions": total_params_millions,
                    "summary/training_time_s": total_training_time,
                    "summary/gpu_name": gpu_name,
                    "summary/gpu_memory_MB": total_memory,
                    "summary/gpu_cores": total_cores,
                    "summary/interrupted": interrupted,
                },
                # Run-level summary, not per step
                step=None,
            )

        self.logger.info(
            f"Training summary logged to MetricTracker for model '{model_name}' on dataset '{dataset_name}'."
        )

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        """
        Encode a list of queries and convert them to a padded tensor.
        Returns:
            - Tensor of shape (batch_size, max_sequence_len)
            - Total number of batches
        """
        if not queries:
            return torch.empty(0, 0, dtype=torch.long), 0
        # Tokenize queries
        tokens = [self.tokenizer.encode(query) for query in queries]
        max_seq_len = self.config.agent.max_response_tokens
        # Pad or truncate to max_seq_len
        padded_tokens = []
        for token_seq in tokens:
            if len(token_seq) > max_seq_len:
                token_seq = token_seq[:max_seq_len]
            else:
                token_seq += [self.tokenizer.pad_token_id] * (
                    max_seq_len - len(token_seq)
                )
            padded_tokens.append(token_seq)
        # Convert to tensor
        tensor = torch.tensor(padded_tokens, dtype=torch.long)
        # Compute number of batches
        num_batches = ceil(len(queries) / self.batch_size)
        return tensor, num_batches

    def profile_training(self, num_steps=20, log_dir: Optional[str] = None):
        """
        Create an AgentProfiler and run the training profiler.
        """
        from src.agent_profiler import AgentProfiler

        profiler = AgentProfiler(self, log_dir=log_dir)
        return profiler.profile_training(num_steps=num_steps)
