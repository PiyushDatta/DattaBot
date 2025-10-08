import os
import shutil
from typing import Optional

import torch

from src.logger import DattaBotLoggerWrapper, get_logger
from src.model import DattaBotModel
from torch import nn


def save_agent(
    model: DattaBotModel,
    filename: str,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save model weights, loss function, optimizer, and optional metadata.

    Args:
        model: The model to save
        filename: File name to save the weights
        loss_fn: Optional AdaptiveLogSoftmaxWithLoss to save
        optimizer: Optional optimizer to save
        metadata: Optional metadata dictionary to save
    """
    logger: DattaBotLoggerWrapper = get_logger()
    temp_filename = None
    logger.info("Saving agent and model...")
    try:
        # Save with temporary file in the same directory to prevent corruption
        # and ensure the rename operation works correctly.
        temp_filename = f"tmp_delete_after_{filename}"
        # Unwrap model if it's wrapped in DP or DDP
        if isinstance(
            model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        # Build checkpoint dictionary
        checkpoint = {
            "model_state_dict": model_state_dict,
            "metadata": metadata or {},
        }
        # Save AdaptiveLogSoftmax weights if provided
        if loss_fn is not None:
            checkpoint["loss_fn_state_dict"] = loss_fn.state_dict()
            logger.debug("Including AdaptiveLogSoftmax weights in checkpoint")
        # Save optimizer state if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            logger.debug("Including optimizer state in checkpoint")
        # Save checkpoint
        torch.save(checkpoint, temp_filename)
        # Atomically move the file to its final destination.
        shutil.move(temp_filename, filename)
        temp_filename = None
        # Log what was saved
        saved_components = ["model"]
        if loss_fn is not None:
            saved_components.append("loss_fn")
        if optimizer is not None:
            saved_components.append("optimizer")
        logger.info(f"Successfully saved {', '.join(saved_components)} to {filename}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        # TODO(PiyushDatta): Fix me at some point though, this is not good.
        # We should do something if we fail saving checkpoint.
    finally:
        # Clean up temp file if it exists
        if temp_filename is not None and os.path.exists(temp_filename):
            os.remove(temp_filename)


def load_agent(
    model: DattaBotModel,
    filepath: str,
    strict: bool = True,
    device: Optional[str] = None,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load model weights, loss function, optimizer, and return metadata.

    Args:
        model: The model to load weights into
        filepath: Path to the weights file
        strict: Whether to enforce strict state dict loading
        device: Optional device to load weights to (defaults to model's current device)
        loss_fn: Optional AdaptiveLogSoftmaxWithLoss to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary containing any metadata saved with the weights
    """
    logger: DattaBotLoggerWrapper = get_logger()
    logger.info("Loading agent and model...")
    try:
        if not os.path.exists(filepath):
            logger.error(f"No checkpoint found, was looking to load from: {filepath}")
            return {}
        # Load checkpoint
        logger.info(f"Checkpoint found at {filepath}, loading...")
        checkpoint = torch.load(filepath, map_location=torch.device(device or "cpu"))
        # Handle structured vs direct state dict (backwards compatibility)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Structured checkpoint
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata", {})
            loss_fn_state = checkpoint.get("loss_fn_state_dict", None)
            optimizer_state = checkpoint.get("optimizer_state_dict", None)
        else:
            # Old format - direct state dict loading
            logger.warning(
                "Loading old checkpoint format (direct state dict). "
                "Consider re-saving with new format."
            )
            state_dict = checkpoint
            metadata = {}
            loss_fn_state = None
            optimizer_state = None
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
        # Load model weights
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            logger.warning(
                f"Model weight loading had incompatible keys:\n"
                f"Missing keys: {incompatible_keys.missing_keys}\n"
                f"Unexpected keys: {incompatible_keys.unexpected_keys}"
            )
        logger.info("Model weights successfully loaded")
        # Load AdaptiveLogSoftmax weights if available
        if loss_fn is not None and loss_fn_state is not None:
            loss_fn.load_state_dict(loss_fn_state)
            logger.debug("AdaptiveLogSoftmax weights successfully loaded")
        elif loss_fn is not None and loss_fn_state is None:
            logger.warning(
                "AdaptiveLogSoftmax provided but no weights found in checkpoint. "
                "Loss function will use random initialization!"
            )
        elif loss_fn_state is not None:
            logger.warning(
                "Checkpoint contains AdaptiveLogSoftmax weights but no loss_fn provided to load into"
            )
        # Load optimizer state if available
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            logger.debug("Optimizer state successfully loaded")
        elif optimizer is not None and optimizer_state is None:
            logger.debug("Optimizer provided but no state found in checkpoint")
        # Log summary
        loaded_components = ["model"]
        if loss_fn is not None and loss_fn_state is not None:
            loaded_components.append("loss_fn")
        if optimizer is not None and optimizer_state is not None:
            loaded_components.append("optimizer")
        logger.info(
            f"Successfully loaded {', '.join(loaded_components)} from {filepath}"
        )
        return metadata
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise
