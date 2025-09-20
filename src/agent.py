import os
import time
import traceback
from typing import Dict, Optional

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from src.agent_config import get_agent_config
from src.api_interface import DattaBotAPIResponse
from src.checkpointing import load_agent, save_agent
from src.communication_mgr import DattaBotCommunicationManager
from src.data_loader import DattabotDataBuilder, DattabotDataLoader

# TODO(PiyushDatta): Get Shampoo optimizer to work.
# from src.optim_shampoo import Shampoo
from src.gpu_profiler import BackgroundGPUProfiler
from src.logger import get_logger
from src.metric_tracker import get_metric_tracker, MetricTracker
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer
from src.util import get_tensor_dtype_from_config, is_device_cpu

from torch import nn, Tensor
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
from torch.utils.data.distributed import DistributedSampler


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.metric_tracker: MetricTracker | None = None
        self.gpu_profiler = None
        # For file names, start with this prefix
        self.agent_fname_prefix = "DATTABOT_VERSION_1_0"
        self.data_dir = self.config.agent.data_directory
        self.plot_dir = self.config.agent.plot_directory
        self.tensor_dtype = get_tensor_dtype_from_config(self.config)
        # Setup hardware settings.
        self.setup_cpu_gpu_settings()
        # Setup multi-gpu settings.
        self.local_rank = 0
        if torch.cuda.device_count() > 1:
            self.setup_distributed()
        # Setup tokenizer.
        self.tokenizer = get_tokenizer(encoding_name="o200k_harmony")
        self.comm_manager = DattaBotCommunicationManager()
        tokenizer_model_name = repr(self.tokenizer)
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        # Setup data loader.
        self.data_builder = DattabotDataBuilder()
        # Setup model.
        self.model = DattaBotModel(device=self.agent_device)
        self.logger.info(f"Model dimensions: {self.config.neural_net.model_dimensions}")
        self.logger.info(
            f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        # Load the model to our device.
        self.model = self.model.to(self.agent_device)
        # Load model weights
        self.weights_fname = f"{self.agent_fname_prefix}_weights.pt"
        load_agent(
            model=self.model,
            filepath=self.weights_fname,
            logger=self.logger,
        )
        # Wrap with distributed layer if we have multiple gpus.
        if torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        self.model.cuda()
        # Setup training objects.
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")
        self.logger.debug(f"Max tokens: {self.config.agent.max_response_tokens}")
        self.lr = self.config.agent.lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.config.agent.weight_decay,
        )
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.lr_scheduler = TorchOneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.config.agent.max_training_num_epochs,
            steps_per_epoch=self.config.agent.num_batches_train_every_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
        )
        # TODO(PiyushDatta): Get Shampoo optimizer to work.
        # self.optimizer = Shampoo(
        #     params=self.model.parameters(),
        #     lr=self.lr,
        #     momentum=0.9,
        #     weight_decay=0.01,
        #     # beta2=0.99,
        #     # block_size=128,
        #     update_freq=1,
        #     epsilon=1e-12
        # )
        # Initialize AMP scaler
        # TODO(PiyushDatta): Turn on AMP scaler after we experiment without it.
        self.scaler = None
        # self.scaler = amp.GradScaler()

    def setup_distributed(self):
        # Ensure this is running under torchrun.
        assert "LOCAL_RANK" in os.environ, (
            "LOCAL_RANK environment variable not found. "
            "You must run using `torchrun --nproc_per_node=N ...` for distributed training. See README for more information."
        )
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.logger.debug(
            f"Setting up setup_distributed with local_rank={self.local_rank}"
        )
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)
        self.agent_device = f"cuda:{self.local_rank}"

    def setup_cpu_gpu_settings(self):
        # Optimize memory allocation
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.agent_device = self.config.env.device
        torch.cuda.empty_cache()

    @property
    def tokenizer_obj(self):
        """
        Accessor for the tokenizer object.
        Example usage: agent.tokenizer_obj.encode(["Hello"])
        """
        return self.tokenizer

    def _is_rank_0(self):
        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

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
        train_dataloader: DattabotDataLoader
        val_dataloader: DattabotDataLoader
        vocab: dict[str, int] = {}
        try:
            self.new_training_session()
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
            self.logger.info("Setting up data.")
            train_dataloader, val_dataloader, vocab = self.data_builder.setup_data()
            # Actual training algorithm.
            self.logger.info(
                f"Got training and validation data for dataset {train_dataloader.dataset_name}. Now going into training loop for "
                f"{self.config.agent.max_training_num_epochs} epochs, "
                f"{num_train_batches_per_phase} batches per training epoch, "
                f"and {num_val_batches_per_phase} batches per validation epoch. "
                f"Length of vocab: {len(vocab)}. "
                f"Batch size: {self.batch_size}."
            )
            for epoch in range(self.config.agent.max_training_num_epochs):
                curr_epoch_num = epoch + 1
                if self._is_rank_0():
                    self.logger.debug(
                        f"\nEpoch {curr_epoch_num}/{self.config.agent.max_training_num_epochs}"
                    )
                # Tell DistributedSampler to shuffle differently this epoch.
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(curr_epoch_num)
                if isinstance(val_dataloader.sampler, DistributedSampler):
                    val_dataloader.sampler.set_epoch(curr_epoch_num)
                # === Training ===
                avg_train_loss, train_steps_this_epoch, train_tokens_in_epoch = (
                    self._train_epoch(
                        epoch_num=curr_epoch_num,
                        dataloader=train_dataloader,
                        num_batches=num_train_batches_per_phase,
                    )
                )
                total_steps += train_steps_this_epoch
                # Update tokens processed
                tokens_processed += train_tokens_in_epoch
                response.num_train_batches += train_steps_this_epoch
                elapsed = time.time() - train_start_time
                if self._is_rank_0():
                    self.logger.debug(
                        f"[Epoch {curr_epoch_num}] avg_train_loss={avg_train_loss:.4f}, "
                        f"tokens_processed={tokens_processed:,}"
                    )
                # Stop conditions
                if tokens_processed >= max_train_tokens:
                    self.logger.info(
                        f"Reached {tokens_processed:,} tokens. Stopping training."
                    )
                    break
                if elapsed >= max_train_time:
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
                response.num_val_batches += val_steps_this_epoch
                if self._is_rank_0():
                    self.logger.debug(
                        f"[Epoch {curr_epoch_num}] avg_val_loss={avg_val_loss:.4f}"
                    )
                    # Log epoch metrics to MetricTracker
                    self.metric_tracker.log_metrics(
                        {
                            "train/loss": avg_train_loss,
                            "val/loss": avg_val_loss,
                        },
                        step=None,
                    )
                # Save best model.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if self._is_rank_0():
                        save_agent(
                            model=self.model,
                            filepath=self.weights_fname,
                            logger=self.logger,
                        )
                        self.logger.info("New best model saved!")
                    self.logger.info(f"Best validation loss: {avg_val_loss:.2f}\n")
                # Periodic checkpoint.
                if self._is_rank_0() and total_steps % ckpt_interval == 0:
                    if self._is_rank_0():
                        save_agent(
                            model=self.model,
                            filepath=self.weights_fname,
                            logger=self.logger,
                        )
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
            if self._is_rank_0():
                save_agent(
                    model=self.model,
                    filepath=self.weights_fname,
                    logger=self.logger,
                )
        # Log the training details, including the time taken to train.
        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time
        # Calculate the total number of model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_millions = total_params // 1_000_000
        self._log_training_details(
            num_train_batches=response.num_train_batches,
            num_val_batches=response.num_val_batches,
            training_score=avg_train_loss,
            validation_score=avg_val_loss,
            dataset_name=train_dataloader.dataset_name,
            model_name=self.agent_fname_prefix,
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
        total_steps = 0
        total_tokens = 0
        log_and_plot_every_x_steps = (
            self.config.agent.logging_and_plotting_every_x_steps
        )
        tgt_vocab_size = self.tokenizer.vocab_size
        # Setup progress bar
        progress_bar = tqdm.tqdm(
            range(num_batches), desc=f"Training for epoch {epoch_num}", leave=True
        )
        # Go through all the batches.
        for _ in progress_bar:
            batch = next(dataloader)
            # [batch, seq_len]
            input_ids, labels = batch[0].to(self.agent_device), batch[1].to(
                self.agent_device
            )
            # Attention mask (1 for real tokens, 0 for pad)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(
                self.agent_device
            )
            # Zero the gradients.
            self.optimizer.zero_grad()
            # Forward + loss, shape: [batch_size, sequence_length, vocab_size]
            with amp.autocast(enabled=(not is_device_cpu(self.agent_device))):
                # Forward pass.
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # [batch*seq_len, vocab]
                logits = logits.view(-1, tgt_vocab_size)
                labels = labels.view(-1)
                loss = self.criterion(logits, labels)
                # If using CrossEntropyLoss, mask out padding tokens.
                padding_mask = labels != self.tokenizer.pad_token_id
                loss = (loss * padding_mask).sum() / padding_mask.sum()
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
            total_steps += 1
            total_tokens += input_ids.numel()
            avg_loss = total_loss / total_steps
            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": avg_loss,
                    "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
                }
            )
            # Log metrics using MetricTracker
            if self._is_rank_0():
                self.metric_tracker.log_metrics(
                    {
                        "train/batch_loss": loss.item(),
                        "train/batch_perplexity": torch.exp(loss).item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=None,
                )
                if total_steps % log_and_plot_every_x_steps == 0:
                    self.logger.info(
                        f"[Epoch {epoch_num} | Step {total_steps}] "
                        f"loss={loss.item():.4f}, "
                        f"perplexity={torch.exp(loss).item():.2f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
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
        tgt_vocab_size = self.tokenizer.vocab_size
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
                attention_mask = (input_ids != self.tokenizer.pad_token_id).to(
                    self.agent_device
                )
                # Forward + loss, shape: [batch_size, sequence_length, vocab_size]
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = logits.view(-1, tgt_vocab_size)
                labels = labels.view(-1)
                # Calculate loss.
                loss = self.criterion(logits, labels)
                padding_mask = labels != self.tokenizer.pad_token_id
                loss = (loss * padding_mask).sum() / padding_mask.sum()
                total_loss += loss.item()
                total_steps += 1
                total_tokens += input_ids.numel()
                avg_loss = total_loss / total_steps
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                # Update progress bar.
                progress_bar.set_postfix(
                    {
                        "val_loss": avg_loss,
                        "val_perplexity": perplexity,
                    }
                )
                if self._is_rank_0():
                    # Log validation batch metrics.
                    self.metric_tracker.log_metrics(
                        {
                            "val/batch_loss": loss.item(),
                            "val/batch_perplexity": torch.exp(loss).item(),
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
        if self._is_rank_0():
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

    def _inference(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        """
        Generate responses for a list of queries using the trained model.
        Args:
            queries: A list of input queries to process.
        Returns:
            A list of DattaBotAPIResponse objects (one per query).
        """
        if not queries:
            self.logger.warning("No queries provided for inference.")
            return [
                DattaBotAPIResponse(
                    response_dict={"output_text": "No queries to process."}
                )
            ]

        self.model.eval()
        responses: list[DattaBotAPIResponse] = []

        try:
            self.logger.info(f"Processing {len(queries)} queries for inference.")
            # Record all user queries in history
            for q in queries:
                self.comm_manager.add_user_message(q)
            # Convert queries to padded tensors
            batch_tensor, _ = self.convert_queries_to_tensors(queries)
            batch_tensor = batch_tensor.to(device=self.agent_device, dtype=torch.long)
            batch_size, _ = batch_tensor.shape

            assert batch_size == len(
                queries
            ), f"Batch size mismatch: {batch_size} != {len(queries)}"

            with torch.no_grad():
                # shape: (batch_size, seq_len, vocab_size)
                logits: Tensor = self.model(batch_tensor)
                # Greedy decoding → (batch_size, seq_len)
                predicted_ids: Tensor = torch.argmax(logits, dim=-1)
                # Decode tokens back into text
                decoded_responses: list[str] = self.tokenizer.decode(
                    tokens_or_tokens_list=predicted_ids.tolist()
                )

            # Build one DattaBotAPIResponse per query
            for i, _ in enumerate(queries):
                reply_text = decoded_responses[i]
                self.comm_manager.add_agent_message(reply_text)
                responses.append(
                    DattaBotAPIResponse(
                        response_dict={"output_text": reply_text},
                        metadata={
                            "tensor_response": predicted_ids[i].clone().detach(),
                            "tokenizer_encodings": batch_tensor[i].tolist(),
                            "tokenizer_decodings": reply_text,
                        },
                    )
                )

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"An error occurred during inference: {e}")
            self.logger.error(f"Traceback:\n{tb_str}")
            error_msg = (
                "An error occurred while processing your request. Please try again."
            )
            self.comm_manager.add_agent_message(error_msg)
            return [DattaBotAPIResponse(response_dict={"output_text": error_msg})]

        return responses

    def respond_to_queries(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        self.logger.info(f"Processing queries: {queries}")
        # Call the inference method
        responses: list[DattaBotAPIResponse] = self._inference(queries=queries)
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert all(
            isinstance(r, DattaBotAPIResponse) for r in responses
        ), "All responses must be DattaBotAPIResponse"
        # Log the results
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
                    "summary/model_name": model_name,
                    "summary/dataset_name": dataset_name,
                    "summary/vocab_length": len(vocab),
                    "summary/batch_size": self.batch_size,
                    "summary/train_batches_completed": num_train_batches,
                    "summary/val_batches_completed": num_val_batches,
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
        # Encode queries using the tokenizer (batch mode).
        tokenized_queries: list[list[int]] = self.tokenizer.encode(queries)
        # Find max sequence length.
        max_seq_len = max(len(seq) for seq in tokenized_queries)
        # Pad each sequence to max_seq_len.
        padded_tensors = [
            (
                torch.tensor(seq, dtype=torch.long)
                if len(seq) == max_seq_len
                else nn.functional.pad(
                    torch.tensor(seq, dtype=torch.long),
                    (0, max_seq_len - len(seq)),
                    value=self.tokenizer.pad_token_id,
                )
            )
            for seq in tokenized_queries
        ]
        # Stack into a single tensor (num_queries, max_seq_len)
        encoded_tensor = torch.stack(padded_tensors, dim=0)
        # Calculate batch info
        total_batch_size = encoded_tensor.size(0)
        num_batches = (total_batch_size + self.batch_size - 1) // self.batch_size
        return encoded_tensor, num_batches

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
