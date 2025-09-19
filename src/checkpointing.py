from typing import Optional
import torch
import os

from src.logger import DattaBotLogger
from src.model import DattaBotModel


def save_agent(model: DattaBotModel, filepath: str, logger: DattaBotLogger) -> None:
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
        # Unwrap model if it's wrapped in DP or DDP
        save_dict = model.state_dict()
        if isinstance(
            model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            save_dict = model.module.state_dict()
        torch.save(save_dict, temp_filepath)
        # Atomic rename
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        logger.info(f"Model weights successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model weights: {str(e)}")
        # We return because we currently do not care if weights fail to save.
        # TODO(PiyushDatta): Fix me at some point though, this is not good.
        return


def load_agent(
    model: DattaBotModel,
    filepath: str,
    logger: DattaBotLogger,
    strict: bool = True,
    device: Optional[str] = None,
) -> dict:
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
            logger.info(
                f"No model weights found, was looking to load weights from: {filepath}"
            )
            return {}
        # Load weights
        logger.info(f"Model weights found at {filepath}, loading weights...")
        checkpoint = torch.load(filepath, map_location=device)
        # Handle structured vs direct state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Load from structured checkpoint
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
        else:
            # Direct state dict loading
            state_dict = checkpoint
            metadata = {}

        # Strip 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            logger.info(
                "Detected DataParallel/DistributedDataParallel checkpoint, "
                "removing 'module.' prefixes from keys..."
            )
            state_dict = {
                k[len("module.") :] if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }

        # Load into model
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)

        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            logger.warning(
                f"Weight loading had incompatible keys:\n"
                f"Missing keys: {incompatible_keys.missing_keys}\n"
                f"Unexpected keys: {incompatible_keys.unexpected_keys}"
            )

        logger.info(f"Model weights successfully loaded from {filepath}")
        return metadata

    except Exception as e:
        logger.error(f"Error loading model weights: {str(e)}")
        raise
