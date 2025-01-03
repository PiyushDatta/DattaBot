import os
import csv
import time
from typing import Dict, Optional
import torch
from torch import Tensor, nn
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.logger import get_logger
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.data_loader import DattabotDataLoader, DattabotDataBuilder
from src.model import DattaBotModel
from src.api_interface import DattaBotAPIResponse
from src.util import get_tensor_dtype_from_config
from src.optim_shampoo import Shampoo
from src.gpu_profiler import BackgroundGPUProfiler


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        # For file names, start with this prefix
        self.agent_fname_prefix = "DATTABOT_VERSION_1_0"
        self.data_dir = self.config.agent.data_directory
        self.plot_dir = self.config.agent.plot_directory
        # Device (gpu or cpu or other)
        self.agent_device = self.config.env.device
        # Setup tokenizer.
        tokenizer_model_name = "distilbert-base-uncased"
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name
        )
        # Add bos and eos tokens and ids
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.bos_token
        )
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.eos_token
        )
        # Things go bad when pad_id is -1.
        # Pad token is -1, change it to eos token.
        assert self.tokenizer.pad_token_id != -1, f"Pad id can't be -1."
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        # Setup data loader.
        self.data_builder = DattabotDataBuilder(tokenizer=self.tokenizer)
        # Setup model.
        self.model = DattaBotModel(tokenizer=self.tokenizer)
        # Batch size.
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")
        # Model dimensions.
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.logger.debug(f"Model dimensions: {self.model_dimensions}")
        # Max tokens for response.
        self.response_max_response_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Max tokens: {self.response_max_response_tokens}")
        # Load model weights
        self.weights_fname = f"{self.agent_fname_prefix}_weights.pt"
        self.load_agent(filepath=self.weights_fname)
        self.lr = self.config.agent.lr
        # With ignore_index=pad_token_id - padding tokens should not contribute to loss.
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.config.agent.weight_decay,
        )
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.writer = SummaryWriter(os.path.join(self.data_dir, "runs"))
        self.logger.info(f"Tensorboard logs will be saved to: {self.writer.log_dir}")
        self.logger.info(f"View logs with: tensorboard --logdir={self.writer.log_dir}")
        # GPU profiler
        self.gpu_profiler = BackgroundGPUProfiler(
            device=self.agent_device,
            sample_every_x_seconds=1.0,
            log_dir=os.path.join(self.data_dir, "gpu_metrics"),
        )
        self.lr_scheduler = TorchOneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.config.env.training_num_epochs,
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
        self.scaler = None
        # Setup GPU settings/optimizations if we have a GPU.
        if not is_device_cpu(self.agent_device):
            self.setup_gpu_settings()

    def setup_gpu_settings(self):
        # Initialize AMP scaler
        self.scaler = amp.GradScaler()
        # Optimize memory allocation
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Adjust batch size and optimization parameters
        self.gradient_accumulation_steps = 1
        # Move model to GPU earlier and optimize
        self.model = self.model.to(self.agent_device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def save_agent(self, filepath: str) -> None:
        """
        Save model weights and optional metadata.

        Args:
            filepath: Path to save the weights
        """
        try:
            # Only create directory if filepath contains a directory path
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            # Save with temporary file to prevent corruption
            temp_filepath = "tmp_delete_after_" + filepath
            save_dict = self.model.state_dict()
            torch.save(save_dict, temp_filepath)
            # Atomic rename
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            self.logger.info(f"Model weights successfully saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model weights: {str(e)}")
            # We return because we currently do not care if weights fail to save.
            # TODO(PiyushDatta): Fix me at some point though, this is not good.
            return

    def load_agent(
        self, filepath: str, strict: bool = True, device: Optional[str] = None
    ) -> Dict:
        """
        Load model weights and return metadata.

        Args:
            filepath: Path to the weights file
            strict: Whether to enforce strict state dict loading
            device: Optional device to load weights to (defaults to model's current device)

        Returns:
            Dictionary containing any metadata saved with the weights
        """
        try:
            if not os.path.exists(filepath):
                self.logger.info(
                    f"No model weights found, was looking to load weights from: {filepath}"
                )
                return
            # Load weights
            checkpoint = torch.load(filepath, map_location=self.agent_device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Load from structured checkpoint
                state_dict = checkpoint["model_state_dict"]
                metadata = checkpoint.get("metadata", {})
            else:
                # Direct state dict loading
                state_dict = checkpoint
                metadata = {}

            # Load weights into model
            incompatible_keys = self.model.load_state_dict(state_dict, strict=strict)

            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                self.logger.warning(
                    f"Weight loading had incompatible keys:\n"
                    f"Missing keys: {incompatible_keys.missing_keys}\n"
                    f"Unexpected keys: {incompatible_keys.unexpected_keys}"
                )

            self.logger.info(f"Model weights successfully loaded from {filepath}")
            return metadata

        except Exception as e:
            self.logger.error(f"Error loading model weights: {str(e)}")
            raise

    def new_training_session(self):
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.gpu_profiler.start()

    def end_training_session(self):
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.gpu_profiler.stop()

    def train_agent(self) -> DattaBotAPIResponse:
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        response.query_response = "Starting training."
        self.logger.info("Started training")
        train_interrupted = False
        avg_train_loss = 0
        avg_val_loss = 0
        train_dataloader: DattabotDataLoader
        val_dataloader: DattabotDataLoader
        vocab: dict[str, int] = {}
        try:
            self.new_training_session()
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
                f"{self.config.env.training_num_epochs} epochs, "
                f"{num_train_batches_per_phase} batches per training epoch, "
                f"and {num_val_batches_per_phase} batches per validation epoch. "
                f"Length of vocab: {len(vocab)}. "
                f"Batch size: {self.batch_size}."
            )
            for epoch in range(self.config.env.training_num_epochs):
                curr_epoch_num = epoch + 1
                self.logger.debug(
                    f"\nEpoch {curr_epoch_num}/{self.config.env.training_num_epochs}"
                )
                # Perform training phase
                avg_train_loss = self._train_epoch(
                    epoch_num=curr_epoch_num,
                    dataloader=train_dataloader,
                    num_batches=num_train_batches_per_phase,
                )
                response.num_train_batches += num_train_batches_per_phase
                self.logger.debug(
                    f"Average train loss after {num_train_batches_per_phase} batches: {avg_train_loss:.2f}"
                )
                # Validate.
                avg_val_loss = self._val_epoch(
                    epoch_num=curr_epoch_num,
                    dataloader=val_dataloader,
                    num_batches=num_val_batches_per_phase,
                )
                response.num_val_batches += num_val_batches_per_phase
                self.logger.debug(
                    f"Average validation loss after {num_val_batches_per_phase} batches: {avg_val_loss:.2f}"
                )
                self.writer.add_scalars(
                    "Loss",
                    {"train": avg_train_loss, "validation": avg_val_loss},
                    curr_epoch_num,
                )
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_agent(filepath=self.weights_fname)
                    self.logger.info("New best model saved!")
                    self.logger.info(
                        f"Epoch {curr_epoch_num}/{self.config.env.training_num_epochs}"
                    )
                    self.logger.info(f"Best validation loss: {avg_val_loss:.2f}\n")
            # Done training!
            response.query_response = (
                f"Successfully trained on {response.num_train_batches} batches. Validated on {response.num_val_batches} batches."
                f"View training progress at: tensorboard --logdir={self.writer.log_dir}"
            )
        except KeyboardInterrupt:
            # Stopped training because we cancelled it via ctrl+c (KeyboardInterrupt).
            train_interrupted = True
            err_msg = f"Training interrupted by user (ctrl+c)."
            response.query_response = err_msg
            self.logger.error(err_msg)
        except Exception as e:
            # Stopped training.
            err_msg = f"Stopped training! Something unexpected happened. Error:\n{e}"
            response.query_response = err_msg
            self.logger.error(err_msg)
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
        self, epoch_num: int, dataloader: DattabotDataLoader, num_batches: int
    ) -> float:
        self.model.train()
        total_loss = 0
        total_steps = 0
        log_and_plot_every_x_steps = (
            self.config.agent.logging_and_plotting_every_x_steps
        )
        tgt_vocab_size = self.tokenizer.vocab_size
        # Setup progress bar
        progress_bar = tqdm.tqdm(
            range(num_batches), desc=f"Training for epoch {epoch_num}", leave=True
        )
        # Go through all the batches.
        for batch_idx in progress_bar:
            batch = next(dataloader)
            # Unpack the batch - batch is a list of [src_data, tgt_data]
            src_data, tgt_data = batch[0], batch[1]
            # Move tensors to device
            src_data = src_data.to(self.agent_device)
            tgt_data = tgt_data.to(self.agent_device)
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (src_data != self.tokenizer.pad_token_id).to(
                self.agent_device
            )
            # Create causal mask for decoder
            seq_length = src_data.size(1)
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), device=self.agent_device),
                diagonal=1,
            ).bool()
            # Zero the gradients.
            self.optimizer.zero_grad()
            # Model predicts.
            # shape: [batch_size, sequence_length, vocab_size]
            with amp.autocast(enabled=(not is_device_cpu(self.agent_device))):
                output_logits = self.model(
                    src_input=src_data,
                    src_mask=attention_mask,
                    tgt_input=tgt_data,
                    tgt_mask=causal_mask,
                )
                # Validate shapes before loss computation
                vocab_dim = output_logits.size(-1)
                assert (
                    vocab_dim == tgt_vocab_size
                ), f"Expected vocab size {tgt_vocab_size}, but got {vocab_dim}."
                # Forward and backward pass.
                loss = self.criterion(
                    # reshape to [batch_size*seq_len, vocab_size]
                    # output_logits.contiguous().view(-1, tgt_vocab_size.size(-1)),
                    output_logits.contiguous().view(-1, tgt_vocab_size),
                    # reshape to [batch_size*seq_len]
                    tgt_data.view(-1),
                )
                # If using CrossEntropyLoss, mask out padding tokens.
                padding_mask = tgt_data.view(-1) != self.tokenizer.pad_token_id
                loss = (loss * padding_mask).sum() / padding_mask.sum()
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
            # Update learning rate if using a scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # Update metrics
            total_loss += loss.item()
            total_steps += 1
            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": total_loss / total_steps,
                    "perplexity": torch.exp(torch.tensor(total_loss / total_steps)),
                }
            )
            if total_steps % log_and_plot_every_x_steps == 0:
                self.logger.info(
                    f"Logging every {log_and_plot_every_x_steps} steps:\n"
                    + str(
                        {
                            "epoch": epoch_num,
                            "train/loss": loss.item(),
                            "train/perplexity": torch.exp(loss).item(),
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )
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
                    # Log the GPU metrics to TensorBoard.
                    self.writer.add_scalar(
                        "GPU/memory_allocated_mb",
                        latest_metrics.memory_allocated,
                        epoch_num,
                    )
                    self.writer.add_scalar(
                        "GPU/utilization_percent", latest_metrics.utilization, epoch_num
                    )
                    self.writer.add_scalar(
                        "GPU/cpu_percent", latest_metrics.cpu_percent, epoch_num
                    )
                    self.writer.add_scalar(
                        "GPU/ram_percent", latest_metrics.ram_percent, epoch_num
                    )
                # Log training details to Tensorboard.
                self.writer.add_scalar(
                    "train/batch_loss",
                    loss.item(),
                    epoch_num * len(dataloader) + batch_idx,
                )
                self.writer.add_scalar(
                    "train/batch_perplexity",
                    torch.exp(loss).item(),
                    epoch_num * len(dataloader) + batch_idx,
                )
                self.writer.add_scalar(
                    "train/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    epoch_num * len(dataloader) + batch_idx,
                )
        # Calculate and return final average loss for epoch.
        # This is the avg_loss.
        return total_loss / total_steps

    def _val_epoch(
        self, epoch_num: int, dataloader: DattabotDataLoader, num_batches: int
    ) -> float:
        self.model.eval()
        total_loss = 0
        total_steps = 0
        tgt_vocab_size = self.tokenizer.vocab_size
        # Setup progress bar
        progress_bar = tqdm.tqdm(
            range(num_batches), desc=f"Validating for epoch {epoch_num}", leave=True
        )
        # Go through all the batches.
        # Disable gradient calculations and go through all the batches.
        with torch.no_grad():
            for batch_idx in progress_bar:
                batch = next(dataloader)
                # Unpack the batch - batch is a list of [src_data, tgt_data]
                src_data, tgt_data = batch[0], batch[1]
                # Move tensors to device
                src_data = src_data.to(self.agent_device)
                tgt_data = tgt_data.to(self.agent_device)
                # Model predicts.
                # shape: [batch_size, sequence_length, vocab_size]
                output_logits = self.model(
                    src_input=src_data, src_mask=None, tgt_input=tgt_data, tgt_mask=None
                )
                # Calculate loss.
                loss = self.criterion(
                    # reshape to [batch_size*seq_len, vocab_size]
                    output_logits.contiguous().view(-1, tgt_vocab_size),
                    # reshape to [batch_size*seq_len]
                    tgt_data.view(-1),
                )
                # Update metrics
                total_loss += loss.item()
                total_steps += 1
                avg_loss = total_loss / total_steps
                perplexity = torch.exp(torch.tensor(avg_loss))
                # Log batch metrics to TensorBoard
                self.writer.add_scalar(
                    "validation/batch_loss",
                    loss.item(),
                    batch_idx + (epoch_num * num_batches),
                )
                self.writer.add_scalar(
                    "validation/batch_perplexity",
                    perplexity.item(),
                    batch_idx + (epoch_num * num_batches),
                )
                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "val_loss": avg_loss,
                        "val_perplexity": perplexity.item(),
                    }
                )
                # Log validation metrics to tensorboard.
                self.writer.add_scalar("validation/loss", avg_loss, epoch_num)
                self.writer.add_scalar(
                    "validation/perplexity",
                    torch.exp(torch.tensor(avg_loss)).item(),
                    epoch_num,
                )
        self.logger.info(
            f"Logging every validation, current epoch {epoch_num}:\n"
            + str(
                {
                    "val/loss": avg_loss,
                    "val/perplexity": torch.exp(torch.tensor(avg_loss)).item(),
                }
            )
        )
        return avg_loss

    def _inference(self, queries: list[str]) -> DattaBotAPIResponse:
        """
        Generate responses for a list of queries using the trained model.
        Args:
            queries: A list of input queries to process.
        Returns:
            A DattaBotAPIResponse containing the model's responses.
        """
        # Validate input
        if not queries or len(queries) == 0:
            self.logger.warning("No queries provided for inference.")
            return DattaBotAPIResponse(query_response="No queries to process.")

        self.model.eval()
        response = DattaBotAPIResponse()
        response.query_response = "Starting inference."
        try:
            # Log the incoming queries.
            self.logger.info(f"Processing {len(queries)} queries for inference.")
            with torch.no_grad():
                for query in queries:
                    for _ in range(self.response_max_response_tokens):
                        # Tokenize the query.
                        src_input = self.tokenizer.encode(
                            query, return_tensors="pt"
                        ).to(self.agent_device)
                        tgt_input = torch.tensor(
                            [[self.tokenizer.bos_token_id]], device=self.agent_device
                        )
                        generated_sequence = []
                        # Pass through the model to get logits.
                        output_logits = self.model(
                            src_input=src_input,
                            src_mask=None,
                            tgt_input=tgt_input,
                            tgt_mask=None,
                        )
                        # 1. Get the predicted token IDs.
                        # Should be shape: [1, 4]
                        # Apply temperature scaling
                        predicted_ids = output_logits.argmax(dim=-1)
                        # 2. Since we want to decode all predicted token IDs,
                        # we need to squeeze the batch dimension (if there's only one batch)
                        # Should be shape: [4]
                        predicted_ids = predicted_ids.squeeze(0)
                        # Check for EOS token
                        if self.tokenizer.eos_token_id in predicted_ids.tolist():
                            break
                        # Update tgt_input with the latest predictions for next iteration
                        tgt_input = predicted_ids.unsqueeze(0)
                    # Decode the generated sequence of token IDs to get the output text
                    predicted_text = self.tokenizer.decode(
                        predicted_ids, skip_special_tokens=True
                    )
                    generated_sequence.append(predicted_text)
            # Convert tensor responses to API response format
            response.query_response = "Inference completed."
            response.query_response = " ".join(generated_sequence)
            self.logger.debug(f"Generated response - {response}")
        except Exception as e:
            self.logger.error(f"An error occurred during inference: {e}")
            response.query_response = (
                "An error occurred while processing your request. Please try again."
            )
        return response

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
        # Create the directory if it doesn't exist
        log_file_path = "training_log.txt"
        # Create or open the log file and write the training details in CSV format.
        with open(log_file_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            gpu_name, total_memory, total_cores = self._get_gpu_info()
            # Write header if the file is empty
            if log_file.tell() == 0:
                writer.writerow(
                    [
                        "Model Name",
                        "Dataset Name",
                        "Vocab Length",
                        "Batch Size",
                        "Training Batches Completed",
                        "Validation Batches Completed",
                        "Training Score",
                        "Validation Score",
                        "Total model parameters (millions)",
                        "Training Time (s)",
                        "Gpu Name",
                        "Total GPU Memory (MB)",
                        "Total GPU Cores",
                        "Interrupted",
                    ]
                )
            # Write the training details.
            writer.writerow(
                [
                    model_name,
                    dataset_name,
                    len(vocab),
                    self.batch_size,
                    num_train_batches,
                    num_val_batches,
                    training_score,
                    validation_score,
                    total_params_millions,
                    total_training_time,
                    gpu_name,
                    total_memory,
                    total_cores,
                    interrupted,
                ]
            )
        self.logger.info(f"Training details logged to {log_file_path}")

    def respond_to_queries(self, queries: list[str]) -> DattaBotAPIResponse:
        self.logger.info(f"Processing queries: {queries}")
        # Call the inference method
        response = self._inference(queries=queries)
        # Log the results
        self.logger.debug(f"Query Response: {response.query_response}")
        self.logger.debug(f"Number of Batches: {response.num_batches}")
        self.logger.debug(f"Tensor Response: {response.tensor_response}")
        return response

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return self.data_builder.tokenizer_encode(decoded_queries=decoded_queries)

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return self.data_builder.tokenizer_decode(encoded_queries=encoded_queries)

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        return self.data_builder.convert_queries_to_tensors(queries=queries)

    def convert_tensors_to_responses(
        self, tensor_resps: list[Tensor]
    ) -> DattaBotAPIResponse:
        return self.data_builder.convert_tensors_to_responses(tensor_resps=tensor_resps)

    def _get_gpu_info(self) -> list[str]:
        if torch.cuda.is_available() and self.agent_device != "cpu":
            gpu_name = torch.cuda.get_device_name(self.agent_device)
            # Convert bytes to MB
            total_memory = torch.cuda.get_device_properties(
                self.agent_device
            ).total_memory // (1024**2)
            total_cores = torch.cuda.get_device_properties(
                self.agent_device
            ).multi_processor_count
        else:
            gpu_name = "Could not retrieve gpu_name"
            total_memory = "Could not retrieve total_memory"
            total_cores = "Could not retrieve total_cores"
        return gpu_name, total_memory, total_cores


def is_device_cpu(agent_device: str):
    return agent_device != "cpu"
