import os
from typing import Dict, Optional
import torch
from torch import Tensor, nn
import tqdm
from transformers import AutoTokenizer
from os.path import isfile

from src.logger import get_logger
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.data_loader import DataLoader
from src.model import DattaBotModel
from src.util import DattaBotAPIResponse, get_tensor_dtype_from_config


class Agent:
    def __init__(self) -> None:
        # For file names, start with this prefix
        self.agent_fname_prefix = "DATTABOT_VERSION_1_0_"
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
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
        # Things go bad when pad_id is -1, we can't even print an encoded tensor!
        # Pad token is -1, change it to eos token.
        assert self.tokenizer.pad_token_id != -1, f"Pad id can't be -1."
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        # Setup data loader.
        self.data_loader = DataLoader(tokenizer=self.tokenizer)
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

            # Determine device for loading
            load_device = self.config.env.device
            # Load weights
            checkpoint = torch.load(filepath, map_location=load_device)
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

    def train_agent(self) -> DattaBotAPIResponse:
        self.model.train()
        epoch_loss = 0
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        # Actual training algorithm.
        best_val_loss = 100
        train_losses = []
        val_losses = []
        for epoch in range(self.config.env.training_num_epochs):
            self.logger.debug(
                f"\nEpoch {epoch + 1}/{self.config.env.training_num_epochs}"
            )
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            val_loss = self.val_epoch()
            val_losses.append(val_loss)
            self.logger.debug(f"Train Loss: {train_loss:.2f}")
            self.logger.debug(f"Validation Loss: {val_loss:.2f}")
            # Save the best model.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_agent(filepath=self.weights_fname)
                self.logger.info("New best model saved!")
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.env.training_num_epochs}"
                )
                self.logger.info(f"Train Loss: {train_loss:.2f}")
                self.logger.info(f"Validation Loss: {val_loss:.2f}\n")
            response.num_batches += 1
            break

        response.query_response = "TODO(PIYUSHDATTA): Fix me. Training agent."
        return response

    def train_epoch(self):
        total_loss = 0
        total_steps = 0
        train_dataloader, self.val_dataloader, self.vocab = (
            self.data_loader.setup_data()
        )
        # Setup progress bar
        progress_bar = tqdm.tqdm(train_dataloader, desc="Training", leave=True)
        # Go through all the batches.
        for batch in progress_bar:
            print("First batch:", batch)
            break
        return 0.5

    def val_epoch(self):
        return 0.5

    def respond_to_queries(self, queries: list[str]) -> DattaBotAPIResponse:
        # Encode the list of queries and convert them into a tensors.
        # Tensor, int
        input_tensor, total_batches = self.convert_queries_to_tensors(queries=queries)
        self.logger.debug(
            f"Output of self.convert_queries_to_tensors():\n{input_tensor}\nwith shape: {input_tensor.shape}\nTotal batches: {total_batches}"
        )
        # Feed the tensors to our model, in batches.
        batched_tensor_responses = []
        tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        total_loss = 0
        for batch, i in enumerate(range(0, total_batches)):
            data, targets = self.data_loader.get_batch(
                input_tensor=input_tensor, idx=i, train=False
            )
            self.logger.debug(
                f"Data being fed to model:\n{data}\nwith shape: {data.shape}"
            )
            # Call our model.
            output = self.model(src_input=data, src_mask=None)
            # Apply softmax to convert logits to probabilities.
            softmax_output = nn.functional.softmax(output, dim=-1).to(tensor_dtype)
            # Flatten and add to our responses.
            output_flat = softmax_output.view(-1, self.response_max_response_tokens)
            batched_tensor_responses.append(output_flat)
            break
        # Decode the tensors and return a DattaBot API response for each tensor.
        response: DattaBotAPIResponse = self.convert_tensors_to_responses(
            tensor_resps=batched_tensor_responses
        )
        self.logger.debug(f"Output of self.respond_to_queries():\n{response}")
        return response

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return self.data_loader.tokenizer_encode(decoded_queries=decoded_queries)

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return self.data_loader.tokenizer_decode(encoded_queries=encoded_queries)

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        return self.data_loader.convert_queries_to_tensors(queries=queries)

    def convert_tensors_to_responses(
        self, tensor_resps: list[Tensor]
    ) -> DattaBotAPIResponse:
        return self.data_loader.convert_tensors_to_responses(tensor_resps=tensor_resps)
