import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from src.logger import DattaBotLoggerWrapper, get_logger
from src.model import DattaBotModel
from src.util import dist_barrier, is_rank_0
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule


@dataclass
class CheckpointComponents:
    """Container for checkpoint components."""

    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    loss_fn_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# FSDP Detection and Options
# ============================================================================


def _is_fsdp_model(model: nn.Module) -> bool:
    """Check if model is FSDP-wrapped."""
    return isinstance(model, FSDPModule) or hasattr(model, "fully_sharded")


def _get_fsdp_state_dict_options(
    full_state_dict: bool = True,
    cpu_offload: bool = True,
    broadcast_from_rank0: bool = False,
) -> StateDictOptions:
    """Create FSDP state dict options."""
    return StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=broadcast_from_rank0,
    )


# ============================================================================
# State Dict Extraction
# ============================================================================


def _extract_model_state(
    model: DattaBotModel,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract model state dict, handling FSDP if necessary."""
    logger = get_logger()

    if _is_fsdp_model(model):
        logger.debug("Extracting FSDP model state (all ranks participating)")
        dist_barrier(device=device)
        options = _get_fsdp_state_dict_options(cpu_offload=True)
        state_dict = get_model_state_dict(model=model, options=options)
        dist_barrier(device=device)
        return state_dict
    else:
        logger.debug("Extracting standard model state")
        unwrapped_model = _unwrap_model(model)
        return unwrapped_model.state_dict()


def _extract_optimizer_state(
    model: DattaBotModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract optimizer state dict, handling FSDP if necessary."""
    logger = get_logger()

    if _is_fsdp_model(model):
        logger.debug("Extracting FSDP optimizer state (all ranks participating)")
        dist_barrier(device=device)
        options = _get_fsdp_state_dict_options(cpu_offload=True)
        state_dict = get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=options,
        )
        dist_barrier(device=device)
        return state_dict
    else:
        logger.debug("Extracting standard optimizer state")
        return optimizer.state_dict()


# ============================================================================
# State Dict Loading
# ============================================================================


def _load_model_state(
    model: DattaBotModel,
    state_dict: Dict[str, Any],
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load model state dict, handling FSDP if necessary."""
    logger = get_logger()
    state_dict = _clean_state_dict_keys(state_dict, model)

    if _is_fsdp_model(model):
        logger.debug("Loading FSDP model state (all ranks participating)")
        dist_barrier(device=device)
        options = _get_fsdp_state_dict_options(broadcast_from_rank0=True)
        set_model_state_dict(
            model=model,
            model_state_dict=state_dict,
            options=options,
        )
        dist_barrier(device=device)
    else:
        logger.debug("Loading standard model state")
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            logger.warning(
                f"Incompatible keys found:\n"
                f"  Missing: {incompatible_keys.missing_keys}\n"
                f"  Unexpected: {incompatible_keys.unexpected_keys}"
            )


def _load_optimizer_state(
    model: DattaBotModel,
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> None:
    """Load optimizer state dict, handling FSDP if necessary."""
    logger = get_logger()

    if _is_fsdp_model(model):
        logger.debug("Loading FSDP optimizer state (all ranks participating)")
        dist_barrier(device=device)
        options = _get_fsdp_state_dict_options(broadcast_from_rank0=True)
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=state_dict,
            options=options,
        )
        dist_barrier(device=device)
    else:
        logger.debug("Loading standard optimizer state")
        optimizer.load_state_dict(state_dict)


# ============================================================================
# Checkpoint Structure Management
# ============================================================================


def _create_checkpoint_dict(components: CheckpointComponents) -> Dict[str, Any]:
    """Create structured checkpoint dictionary."""
    checkpoint = {
        "model_state_dict": components.model_state,
        "metadata": components.metadata or {},
    }

    if components.optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = components.optimizer_state

    if components.loss_fn_state is not None:
        checkpoint["loss_fn_state_dict"] = components.loss_fn_state

    return checkpoint


def _parse_checkpoint_dict(checkpoint: Dict[str, Any]) -> CheckpointComponents:
    """Parse checkpoint dictionary, handling backwards compatibility."""
    logger = get_logger()

    # New structured format
    if "model_state_dict" in checkpoint:
        return CheckpointComponents(
            model_state=checkpoint["model_state_dict"],
            optimizer_state=checkpoint.get("optimizer_state_dict"),
            loss_fn_state=checkpoint.get("loss_fn_state_dict"),
            metadata=checkpoint.get("metadata", {}),
        )

    # Legacy format (direct state dict)
    logger.warning(
        "Loading legacy checkpoint format. Consider re-saving with current version."
    )
    return CheckpointComponents(
        model_state=checkpoint,
        metadata={},
    )


def _clean_state_dict_keys(
    state_dict: Dict[str, Any], model: nn.Module
) -> Dict[str, Any]:
    """
    Clean state dict keys to match target model wrapper type.

    Handles conversion between:
    - DDP/DataParallel checkpoint -> regular model (remove 'module.' prefix)
    - Regular checkpoint -> DDP/DataParallel model (add 'module.' prefix)
    - Matching types (no conversion needed)

    Args:
        state_dict: State dictionary to clean
        model: Target model to load into

    Returns:
        Cleaned state dictionary with appropriate prefixes
    """
    logger = get_logger()

    # Check if checkpoint has 'module.' prefix
    checkpoint_has_prefix = any(k.startswith("module.") for k in state_dict.keys())

    # Check if target model is DDP/DataParallel wrapped
    model_is_wrapped = isinstance(
        model,
        (nn.DataParallel, nn.parallel.DistributedDataParallel),
    )
    # Case 1: Checkpoint has prefix, but model is not wrapped -> Remove prefix
    if checkpoint_has_prefix and not model_is_wrapped:
        logger.info(
            "Checkpoint was saved from DDP/DataParallel model, "
            "removing 'module.' prefix to load into unwrapped model"
        )
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # Case 2: Checkpoint has no prefix, but model is wrapped -> Add prefix
    elif not checkpoint_has_prefix and model_is_wrapped:
        logger.info(
            "Checkpoint was saved from unwrapped model, "
            "adding 'module.' prefix to load into DDP/DataParallel model"
        )
        return {f"module.{k}": v for k, v in state_dict.items()}
    # Case 3 & 4: Prefix matches wrapper type -> No conversion needed
    else:
        if checkpoint_has_prefix and model_is_wrapped:
            logger.debug(
                "Loading DDP/DataParallel checkpoint into wrapped model (no conversion)"
            )
        elif not checkpoint_has_prefix and not model_is_wrapped:
            logger.debug(
                "Loading unwrapped checkpoint into unwrapped model (no conversion)"
            )
        return state_dict


# ============================================================================
# File Operations
# ============================================================================


def _atomic_save(checkpoint: Dict[str, Any], filepath: str) -> None:
    """
    Atomically save checkpoint to disk.

    Uses temporary file + atomic rename to prevent corruption.
    """
    logger = get_logger()
    filepath = Path(filepath)
    temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"

    try:
        # Save to temporary file
        torch.save(checkpoint, temp_filepath)

        # Atomic rename
        shutil.move(str(temp_filepath), str(filepath))

        logger.debug(f"Atomically saved checkpoint to {filepath}")

    except Exception as e:
        # Clean up temp file on error
        if temp_filepath.exists():
            temp_filepath.unlink()
        raise
    finally:
        # Double-check cleanup
        if temp_filepath.exists():
            temp_filepath.unlink()


def _should_save_on_this_rank() -> bool:
    """Determine if current rank should save checkpoint."""
    return not dist.is_available() or not dist.is_initialized() or is_rank_0()


# ============================================================================
# Utility Functions
# ============================================================================


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DDP/DataParallel wrapper."""
    if isinstance(
        model,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
    ):
        return model.module
    return model


def _log_checkpoint_components(
    components: CheckpointComponents,
    operation: str,
    filepath: str,
) -> None:
    """Log which components were saved/loaded."""
    logger = get_logger()

    component_list = ["model"]
    if components.optimizer_state is not None:
        component_list.append("optimizer")
    if components.loss_fn_state is not None:
        component_list.append("loss_fn")

    logger.info(
        f"Successfully {operation} {', '.join(component_list)} "
        f"{'to' if operation == 'saved' else 'from'} {filepath}"
    )


# ============================================================================
# Main Public API
# ============================================================================


def save_agent(
    model: DattaBotModel,
    filename: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model, optimizer, and optional loss function with metadata.

    Supports both standard and FSDP models. Uses atomic file operations
    to prevent corruption. All ranks participate in FSDP state extraction,
    but only rank 0 writes to disk.

    Args:
        model: The model to save
        filename: Destination filepath
        device: Device the model is on
        optimizer: Optimizer to save
        loss_fn: Optional AdaptiveLogSoftmaxWithLoss to save
        metadata: Optional metadata dictionary

    Raises:
        Exception: If save operation fails
    """
    logger = get_logger()
    logger.info(f"Saving checkpoint to {filename}")

    try:
        # Extract state dicts (all ranks participate for FSDP)
        model_state = _extract_model_state(model, device)
        optimizer_state = _extract_optimizer_state(model, optimizer, device)

        # Only rank 0 saves to disk
        if _should_save_on_this_rank():
            components = CheckpointComponents(
                model_state=model_state,
                optimizer_state=optimizer_state,
                loss_fn_state=loss_fn.state_dict() if loss_fn else None,
                metadata=metadata,
            )

            checkpoint = _create_checkpoint_dict(components)
            _atomic_save(checkpoint, filename)
            _log_checkpoint_components(components, "saved", filename)

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {repr(e)}\n{traceback.format_exc()}")
        raise
    finally:
        dist_barrier(device=device)


def load_agent(
    model: DattaBotModel,
    filepath: str,
    optimizer: torch.optim.Optimizer,
    device: str,
    strict: bool = True,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
) -> Dict[str, Any]:
    """
    Load model, optimizer, and optional loss function from checkpoint.

    Supports both standard and FSDP models. Handles backwards compatibility
    with legacy checkpoint formats.

    Args:
        model: Model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optimizer to load state into
        device: Device to load weights to
        strict: Whether to enforce strict state dict loading
        loss_fn: Optional AdaptiveLogSoftmaxWithLoss to load

    Returns:
        Metadata dictionary from checkpoint

    Raises:
        Exception: If load operation fails
    """
    logger = get_logger()
    logger.info(f"Loading checkpoint from {filepath}")

    try:
        # Validate file exists
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint not found at {filepath}")
            return {}

        # Load checkpoint file
        checkpoint = torch.load(filepath, map_location=torch.device(device or "cpu"))
        components = _parse_checkpoint_dict(checkpoint)

        # Load model state (all ranks participate for FSDP)
        _load_model_state(model, components.model_state, device, strict)
        logger.info("Model weights loaded successfully")

        # Load optimizer state if available
        if optimizer is not None:
            if components.optimizer_state is not None:
                _load_optimizer_state(
                    model, optimizer, components.optimizer_state, device
                )
                logger.debug("Optimizer state loaded successfully")
            else:
                logger.debug("No optimizer state found in checkpoint")

        # Load loss function state if available
        if loss_fn is not None:
            if components.loss_fn_state is not None:
                loss_fn.load_state_dict(components.loss_fn_state)
                logger.debug("Loss function state loaded successfully")
            else:
                logger.warning(
                    "Loss function provided but no state in checkpoint. "
                    "Using random initialization!"
                )
        elif components.loss_fn_state is not None:
            logger.warning("Checkpoint contains loss function state but none provided")

        # Log summary
        _log_checkpoint_components(components, "loaded", filepath)

        return components.metadata or {}

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise
    finally:
        dist_barrier(device=device)
