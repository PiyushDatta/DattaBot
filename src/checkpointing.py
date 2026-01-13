import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from safetensors.torch import (
    load_file as load_safetensors,
    save_file as save_safetensors,
)
from src.logger import get_logger
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
# Path Management
# ============================================================================


def _get_checkpoint_paths(base_path: Path) -> Dict[str, Path]:
    """Get paths for all checkpoint components."""
    return {
        "model": base_path / "model.safetensors",
        "optimizer": base_path / "optimizer.pt",
        "loss_fn": base_path / "loss_fn.pt",
        "metadata": base_path / "metadata.json",
    }


# ============================================================================
# SafeTensors Utilities
# ============================================================================


def _flatten_state_dict(
    state_dict: Dict[str, Any], prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """Flatten nested state dict to safetensors-compatible format."""
    flat = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, torch.Tensor):
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_state_dict(value, f"{full_key}."))
    return flat


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return obj


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
# State Dict Key Cleaning
# ============================================================================


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


def _ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _atomic_save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    filepath: Path,
) -> None:
    """
    Atomically save state dict as safetensors.
    Uses temporary file + atomic rename to prevent corruption.
    """
    logger = get_logger()
    temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"
    try:
        # Flatten nested state dict for safetensors compatibility
        flat_state_dict = _flatten_state_dict(state_dict)
        # Ensure all tensors are contiguous and on CPU
        flat_state_dict = {k: v.contiguous().cpu() for k, v in flat_state_dict.items()}
        # Save to temporary file
        save_safetensors(flat_state_dict, str(temp_filepath))
        # Atomic rename
        shutil.move(str(temp_filepath), str(filepath))
        logger.debug(f"Atomically saved safetensors to {filepath}")
    except Exception as e:
        if temp_filepath.exists():
            temp_filepath.unlink()
        logger.error(f"Error saving safetensors to {filepath}: {e}")
        raise
    finally:
        if temp_filepath.exists():
            temp_filepath.unlink()


def _atomic_save_pt(state_dict: Dict[str, Any], filepath: Path) -> None:
    """
    Atomically save state dict as PyTorch .pt file.
    Uses temporary file + atomic rename to prevent corruption.
    """
    logger = get_logger()
    temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"

    try:
        torch.save(state_dict, temp_filepath)
        shutil.move(str(temp_filepath), str(filepath))
        logger.debug(f"Atomically saved .pt to {filepath}")
    except Exception as e:
        if temp_filepath.exists():
            temp_filepath.unlink()
        logger.error(f"Error saving .pt to {filepath}: {e}")
        raise
    finally:
        if temp_filepath.exists():
            temp_filepath.unlink()


def _atomic_save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Atomically save data as JSON file.
    Uses temporary file + atomic rename to prevent corruption.
    """
    logger = get_logger()
    temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"

    try:
        serializable_data = _make_json_serializable(data)
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        shutil.move(str(temp_filepath), str(filepath))
        logger.debug(f"Atomically saved JSON to {filepath}")
    except Exception as e:
        if temp_filepath.exists():
            temp_filepath.unlink()
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise
    finally:
        if temp_filepath.exists():
            temp_filepath.unlink()


def _load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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
    checkpoint_dir: str,
) -> None:
    """Log which components were saved/loaded."""
    logger = get_logger()

    component_list = ["model"]
    if components.optimizer_state is not None:
        component_list.append("optimizer")
    if components.loss_fn_state is not None:
        component_list.append("loss_fn")
    if components.metadata is not None:
        component_list.append("metadata")

    logger.info(
        f"Successfully {operation} {', '.join(component_list)} "
        f"{'to' if operation == 'saved' else 'from'} {checkpoint_dir}"
    )


def _checkpoint_exists(checkpoint_dir: Path) -> bool:
    """Check if a valid checkpoint exists at the given directory."""
    paths = _get_checkpoint_paths(checkpoint_dir)
    return paths["model"].exists()


# ============================================================================
# Main Public API
# ============================================================================


def save_agent(
    model: DattaBotModel,
    checkpoint_dir: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model, optimizer, and optional loss function with metadata.

    Saves components as:
    - model.safetensors: Model weights in safetensors format
    - optimizer.pt: Optimizer state as PyTorch file
    - loss_fn.pt: Loss function state as PyTorch file (if provided)
    - metadata.json: Metadata as JSON file

    Supports both standard and FSDP models. Uses atomic file operations
    to prevent corruption. All ranks participate in FSDP state extraction,
    but only rank 0 writes to disk.

    Args:
        model: The model to save
        checkpoint_dir: Destination directory for checkpoint files
        device: Device the model is on
        optimizer: Optimizer to save
        loss_fn: Optional AdaptiveLogSoftmaxWithLoss to save
        metadata: Optional metadata dictionary

    Raises:
        Exception: If save operation fails
    """
    logger = get_logger()
    checkpoint_path = Path(checkpoint_dir)
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    try:
        # Extract state dicts (all ranks participate for FSDP)
        model_state = _extract_model_state(model, device)
        optimizer_state = _extract_optimizer_state(model, optimizer, device)
        # Only rank 0 saves to disk
        if _should_save_on_this_rank():
            # Ensure checkpoint directory exists
            _ensure_directory(checkpoint_path)
            paths = _get_checkpoint_paths(checkpoint_path)
            # Save model state as safetensors
            _atomic_save_safetensors(model_state, paths["model"])
            # Save optimizer state as .pt
            _atomic_save_pt(optimizer_state, paths["optimizer"])
            # Save loss function state as .pt (if provided)
            if loss_fn is not None:
                _atomic_save_pt(loss_fn.state_dict(), paths["loss_fn"])
            # Save metadata as JSON
            _atomic_save_json(metadata or {}, paths["metadata"])
            components = CheckpointComponents(
                model_state=model_state,
                optimizer_state=optimizer_state,
                loss_fn_state=loss_fn.state_dict() if loss_fn else None,
                metadata=metadata,
            )
            _log_checkpoint_components(components, "saved", str(checkpoint_path))

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {repr(e)}\n{traceback.format_exc()}")
        raise

    finally:
        dist_barrier(device=device)


def load_agent(
    model: DattaBotModel,
    checkpoint_dir: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    strict: bool = True,
    loss_fn: Optional[nn.AdaptiveLogSoftmaxWithLoss] = None,
) -> Dict[str, Any]:
    """
    Load model, optimizer, and optional loss function from checkpoint.

    Expects checkpoint directory containing:
    - model.safetensors: Model weights in safetensors format
    - optimizer.pt: Optimizer state as PyTorch file
    - loss_fn.pt: Loss function state as PyTorch file (optional)
    - metadata.json: Metadata as JSON file

    Supports both standard and FSDP models.

    Args:
        model: Model to load weights into
        checkpoint_dir: Path to checkpoint directory
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
    checkpoint_path = Path(checkpoint_dir)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    try:
        # Validate checkpoint exists
        if not _checkpoint_exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return {}

        paths = _get_checkpoint_paths(checkpoint_path)
        # Load model state from safetensors
        model_state = load_safetensors(str(paths["model"]), device=device.index)
        _load_model_state(model, model_state, device, strict)
        logger.info("Model weights loaded successfully")
        # Load optimizer state if available
        if optimizer is not None and paths["optimizer"].exists():
            optimizer_state = torch.load(
                paths["optimizer"],
                map_location=torch.device(device or "cpu"),
            )
            _load_optimizer_state(model, optimizer, optimizer_state, device)
            logger.debug("Optimizer state loaded successfully")
        elif optimizer is not None:
            logger.debug("No optimizer state found in checkpoint")

        # Load loss function state if available
        if loss_fn is not None:
            if paths["loss_fn"].exists():
                loss_fn_state = torch.load(
                    paths["loss_fn"],
                    map_location=torch.device(device or "cpu"),
                )
                loss_fn.load_state_dict(loss_fn_state)
                logger.debug("Loss function state loaded successfully")
            else:
                logger.warning(
                    "Loss function provided but no state in checkpoint. "
                    "Using random initialization!"
                )
        elif paths["loss_fn"].exists():
            logger.warning("Checkpoint contains loss function state but none provided")

        # Load metadata
        metadata = {}
        if paths["metadata"].exists():
            metadata = _load_json(paths["metadata"])
            logger.debug("Metadata loaded successfully")

        # Log summary
        components = CheckpointComponents(
            model_state=model_state,
            optimizer_state=(
                (optimizer_state if paths["optimizer"].exists() else None)
                if optimizer
                else None
            ),
            loss_fn_state=(
                (loss_fn_state if paths["loss_fn"].exists() else None)
                if loss_fn
                else None
            ),
            metadata=metadata,
        )
        _log_checkpoint_components(components, "loaded", str(checkpoint_path))

        return metadata

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}\n{traceback.format_exc()}")
        raise

    finally:
        dist_barrier(device=device)
