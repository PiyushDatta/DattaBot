"""
Checkpoint management for DattaBot Agent.

Provides a singleton DattaBotCheckpointManager for saving and loading model, optimizer,
and loss function states with atomic file operations and TPU/XLA synchronization.

Usage:
    # Initialize manager (first call configures it)
    manager = get_checkpoint_manager(checkpoint_dir="/path/to/checkpoints")

    # Configure bundle references (once during initialization)
    manager.bundle.unwrapped_model = model
    manager.bundle.optimizer = optimizer
    manager.bundle.loss_fn = loss_fn
    manager.bundle.device = device

    # During training, update state and save
    manager.bundle.update_state(epoch=5, global_step=1000, train_loss=0.5)
    manager.save_agent()

    # Load checkpoint (loads into bundle)
    metadata = manager.load_agent()
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import (
    load_file as load_safetensors,
    save_file as save_safetensors,
)
from src.logger import get_logger
from src.util import dist_barrier, is_rank_0, Singleton
from torch import nn

# TPU support
try:
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CheckpointMetadata:
    """Structured metadata for checkpoints."""

    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    epoch: int = 0
    global_step: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    tokens_processed: int = 0
    agent_name: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create metadata from dictionary."""
        # Handle backwards compatibility for older checkpoints
        known_fields = {
            "version",
            "created_at",
            "epoch",
            "global_step",
            "train_loss",
            "val_loss",
            "tokens_processed",
            "agent_name",
            "extra",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        # Put unknown fields in extra
        extra = filtered.get("extra", {})
        for k, v in data.items():
            if k not in known_fields:
                extra[k] = v
        filtered["extra"] = extra
        return cls(**filtered)


@dataclass
class CheckpointPaths:
    """Paths for checkpoint components."""

    base_dir: Path
    model: Path
    optimizer: Path
    loss_fn: Path
    metadata: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "CheckpointPaths":
        """Create paths from base directory."""
        return cls(
            base_dir=base_dir,
            model=base_dir / "model.safetensors",
            optimizer=base_dir / "optimizer.pt",
            loss_fn=base_dir / "loss_fn.pt",
            metadata=base_dir / "metadata.json",
        )

    def exists(self) -> bool:
        """Check if checkpoint exists (model file present)."""
        return self.model.exists()


@dataclass
class CheckpointBundle:
    """
    Global bundle of references to checkpointable components.

    Holds references to model, optimizer, etc. that are set once during
    initialization. Only primitive state values (epoch, step, loss) get
    updated during training.

    Usage:
        # During initialization (set references once)
        bundle = get_checkpoint_bundle()
        bundle.unwrapped_model = model
        bundle.optimizer = optimizer
        bundle.device = device

        # During training (update state)
        bundle.update_state(epoch=5, global_step=1000, train_loss=0.5)

        # Checkpoint manager uses bundle automatically
        manager.save_agent()
    """

    # References (set once, don't recreate)
    unwrapped_model: Optional[nn.Module] = None
    wrapped_model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    loss_fn: Optional[nn.Module] = None
    lr_scheduler: Optional[Any] = None
    device: Optional[torch.device] = None
    agent_name: str = ""
    # Mutable training state (updated during training)
    epoch: int = 0
    global_step: int = 0
    tokens_processed: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None

    def update_state(
        self,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        tokens_processed: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> "CheckpointBundle":
        """Update mutable training state. Returns self for chaining."""
        if epoch is not None:
            self.epoch = epoch
        if global_step is not None:
            self.global_step = global_step
        if tokens_processed is not None:
            self.tokens_processed = tokens_processed
        if train_loss is not None:
            self.train_loss = train_loss
        if val_loss is not None:
            self.val_loss = val_loss
        return self

    def is_configured(self) -> bool:
        """Check if bundle has required references set."""
        return self.unwrapped_model is not None and self.device is not None


# =============================================================================
# Checkpoint Manager
# =============================================================================


class DattaBotCheckpointManager(metaclass=Singleton):
    """
    Singleton checkpoint manager for saving and loading model states.

    Supports:
    - Standard PyTorch models
    - TPU/XLA devices
    - Atomic file operations
    - Structured metadata tracking
    - Built-in CheckpointBundle for reference tracking

    Usage:
        # Initialize manager (first call)
        manager = get_checkpoint_manager(checkpoint_dir="/path/to/checkpoints")

        # Configure bundle references (once during init)
        manager.bundle.unwrapped_model = model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = device

        # During training, update state and save
        manager.bundle.update_state(epoch=5, global_step=1000, train_loss=0.5)
        manager.save_agent()
    """

    def __init__(self, checkpoint_dir: Path | str | None = None) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files (required on first call)
        """
        self.logger = get_logger()
        # Initialize bundle if not already present (singleton may be reused)
        if not hasattr(self, "bundle"):
            self.bundle = CheckpointBundle()
        # Only configure if checkpoint_dir is provided (first initialization)
        if checkpoint_dir is not None:
            self.configure(checkpoint_dir)
        elif not hasattr(self, "checkpoint_dir"):
            raise ValueError(
                "DattaBotCheckpointManager requires checkpoint_dir on first initialization"
            )

    def configure(self, checkpoint_dir: Path | str) -> "DattaBotCheckpointManager":
        """
        Configure or reconfigure the checkpoint directory.

        Args:
            checkpoint_dir: Directory for checkpoint files

        Returns:
            Self for method chaining
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.paths = CheckpointPaths.from_base_dir(self.checkpoint_dir)
        self.logger.debug(
            f"DattaBotCheckpointManager configured with dir: {self.checkpoint_dir}"
        )
        return self

    # =========================================================================
    # Public API
    # =========================================================================

    def save_agent(self) -> None:
        """
        Save model, optimizer, and loss function from the bundle.

        Only rank 0 writes to disk. Uses atomic file operations to prevent corruption.
        """
        if not self.bundle.is_configured():
            raise ValueError(
                "Bundle not configured. Set bundle.unwrapped_model and bundle.device first."
            )
        self.logger.info(f"Saving checkpoint to {self.checkpoint_dir}")

        try:
            # Sync before saving
            self._sync_device(self.bundle.device)

            # Extract state dicts - ALL ranks must participate in this
            # because DTensor.full_tensor() is a collective operation
            model_state = self.bundle.unwrapped_model.state_dict()
            optimizer_state = (
                self.bundle.optimizer.state_dict()
                if self.bundle.optimizer is not None
                else {}
            )

            # Convert tensors to CPU on ALL ranks (collective operation for DTensor)
            # This must happen before the rank 0 check
            model_state_cpu = {
                k: _to_clean_cpu_tensor(v) for k, v in model_state.items()
            }

            # Only rank 0 saves to disk
            if _should_save_on_this_rank():
                self._ensure_directory(self.checkpoint_dir)
                # Save model as safetensors (already converted to CPU tensors)
                self._atomic_save_safetensors_direct(model_state_cpu, self.paths.model)
                # Save optimizer state
                if optimizer_state:
                    self._atomic_save_pt(optimizer_state, self.paths.optimizer)
                # Save loss function if provided
                if self.bundle.loss_fn is not None:
                    self._atomic_save_pt(
                        self.bundle.loss_fn.state_dict(), self.paths.loss_fn
                    )
                # Build and save metadata
                metadata = CheckpointMetadata(
                    epoch=self.bundle.epoch,
                    global_step=self.bundle.global_step,
                    train_loss=self.bundle.train_loss,
                    val_loss=self.bundle.val_loss,
                    tokens_processed=self.bundle.tokens_processed,
                    agent_name=self.bundle.agent_name,
                )
                self._save_metadata(metadata)
                self.logger.info(
                    f"Checkpoint saved: epoch={self.bundle.epoch}, step={self.bundle.global_step}, "
                    f"train_loss={f'{self.bundle.train_loss:.4f}' if self.bundle.train_loss is not None else 'N/A'}, "
                    f"val_loss={f'{self.bundle.val_loss:.4f}' if self.bundle.val_loss is not None else 'N/A'}"
                )

            # Barrier to ensure all ranks wait for save to complete
            dist_barrier(device=self.bundle.device)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

    def save_agent_fsdp(self) -> None:
        """
        Fast checkpoint saving for FSDP models using sharded state dicts.

        Instead of gathering all shards to rank 0 (slow), each rank saves
        its own shard directly. This is MUCH faster for large models.

        Uses PyTorch's distributed checkpoint API for efficient saving.
        """
        if not self.bundle.is_configured():
            raise ValueError(
                "Bundle not configured. Set bundle.unwrapped_model and bundle.device first."
            )
        self.logger.info(f"Saving FSDP checkpoint to {self.checkpoint_dir}")

        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict,
                StateDictOptions,
            )

            # Sync before saving
            self._sync_device(self.bundle.device)
            self._ensure_directory(self.checkpoint_dir)

            # Get sharded state dict (no all-gather, each rank keeps its shard)
            model_state, optimizer_state = get_state_dict(
                self.bundle.unwrapped_model,
                self.bundle.optimizer,
                options=StateDictOptions(
                    full_state_dict=False,  # Keep sharded, don't gather
                    cpu_offload=True,  # Offload to CPU to save GPU memory
                ),
            )

            # Use distributed checkpoint to save shards in parallel
            state_dict = {
                "model": model_state,
                "optimizer": optimizer_state,
            }

            dcp.save(
                state_dict,
                checkpoint_id=str(self.checkpoint_dir / "distributed_ckpt"),
            )

            # Save metadata (only rank 0)
            if _should_save_on_this_rank():
                # Save loss function if provided
                if self.bundle.loss_fn is not None:
                    self._atomic_save_pt(
                        self.bundle.loss_fn.state_dict(), self.paths.loss_fn
                    )
                # Build and save metadata
                metadata = CheckpointMetadata(
                    epoch=self.bundle.epoch,
                    global_step=self.bundle.global_step,
                    train_loss=self.bundle.train_loss,
                    val_loss=self.bundle.val_loss,
                    tokens_processed=self.bundle.tokens_processed,
                    agent_name=self.bundle.agent_name,
                )
                self._save_metadata(metadata)
                self.logger.info(
                    f"FSDP checkpoint saved: epoch={self.bundle.epoch}, step={self.bundle.global_step}, "
                    f"train_loss={f'{self.bundle.train_loss:.4f}' if self.bundle.train_loss is not None else 'N/A'}, "
                    f"val_loss={f'{self.bundle.val_loss:.4f}' if self.bundle.val_loss is not None else 'N/A'}"
                )

            dist_barrier(device=self.bundle.device)

        except ImportError:
            self.logger.warning(
                "torch.distributed.checkpoint not available, falling back to slow save"
            )
            self.save_agent()
        except Exception as e:
            self.logger.error(f"Failed to save FSDP checkpoint: {e}")
            raise

    def load_agent(self, strict: bool = True) -> CheckpointMetadata:
        """
        Load model, optimizer, and loss function into the bundle.

        Args:
            strict: Whether to enforce strict state dict loading for model
        Returns:
            CheckpointMetadata from the checkpoint
        """
        if self.bundle.unwrapped_model is None:
            raise ValueError("Bundle not configured. Set bundle.unwrapped_model first.")
        self.logger.info(f"Loading checkpoint from {self.checkpoint_dir}")

        try:
            # Validate checkpoint exists
            if not self.paths.exists():
                self.logger.warning(f"No checkpoint found at {self.checkpoint_dir}")
                return CheckpointMetadata()
            # Load model state into unwrapped model
            model_state = load_safetensors(str(self.paths.model), device="cpu")
            model_state = _clean_state_dict_keys(
                model_state, self.bundle.unwrapped_model
            )
            incompatible = self.bundle.unwrapped_model.load_state_dict(
                model_state, strict=strict
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                self.logger.warning(
                    f"Incompatible keys:\n"
                    f"  Missing: {incompatible.missing_keys}\n"
                    f"  Unexpected: {incompatible.unexpected_keys}"
                )
            self.logger.info("Model weights loaded successfully")
            # Load optimizer state if available
            if self.bundle.optimizer is not None:
                self._load_optimizer_state()
            # Load loss function state if available
            if self.bundle.loss_fn is not None:
                self._load_loss_fn_state()
            # Load and return metadata
            metadata = self._load_metadata()
            # Update bundle state from metadata
            self.bundle.epoch = metadata.epoch
            self.bundle.global_step = metadata.global_step
            self.bundle.tokens_processed = metadata.tokens_processed
            self.bundle.train_loss = metadata.train_loss
            self.bundle.val_loss = metadata.val_loss
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
        finally:
            if self.bundle.device is not None:
                dist_barrier(device=self.bundle.device)

    def load_agent_fsdp(self) -> CheckpointMetadata:
        """
        Fast checkpoint loading for FSDP models using sharded state dicts.

        Loads sharded checkpoints saved by save_agent_fsdp(). Each rank loads
        only its own shard, which is much faster than gathering the full model.
        """
        if self.bundle.unwrapped_model is None:
            raise ValueError("Bundle not configured. Set bundle.unwrapped_model first.")

        distributed_ckpt_path = self.checkpoint_dir / "distributed_ckpt"
        if not distributed_ckpt_path.exists():
            self.logger.info(f"No FSDP checkpoint found at {distributed_ckpt_path}")
            return CheckpointMetadata()

        self.logger.info(f"Loading FSDP checkpoint from {distributed_ckpt_path}")

        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict,
                set_state_dict,
                StateDictOptions,
            )

            # Get current sharded state dict structure
            model_state, optimizer_state = get_state_dict(
                self.bundle.unwrapped_model,
                self.bundle.optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )

            state_dict = {
                "model": model_state,
                "optimizer": optimizer_state,
            }

            # Load sharded checkpoint (each rank loads its shard)
            dcp.load(
                state_dict,
                checkpoint_id=str(distributed_ckpt_path),
            )

            # Apply loaded state to model and optimizer
            set_state_dict(
                self.bundle.unwrapped_model,
                self.bundle.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )

            self.logger.info("FSDP checkpoint loaded successfully")

            # Load loss function state if available
            if self.bundle.loss_fn is not None:
                self._load_loss_fn_state()

            # Load and return metadata
            metadata = self._load_metadata()
            self.bundle.epoch = metadata.epoch
            self.bundle.global_step = metadata.global_step
            self.bundle.tokens_processed = metadata.tokens_processed
            self.bundle.train_loss = metadata.train_loss
            self.bundle.val_loss = metadata.val_loss

            return metadata

        except ImportError:
            self.logger.warning(
                "torch.distributed.checkpoint not available, falling back to slow load"
            )
            return self.load_agent()
        except Exception as e:
            self.logger.error(f"Failed to load FSDP checkpoint: {e}")
            raise
        finally:
            if self.bundle.device is not None:
                dist_barrier(device=self.bundle.device)

    def load_model(
        self,
        model: nn.Module,
        device: torch.device,
        target_dtype: torch.dtype | None = None,
    ) -> bool:
        """
        Load checkpoint into a model (non-FSDP).

        This is for loading weights into a model before any distributed
        wrapping is applied.

        Args:
            model: The model to load weights into
            device: The device to move weights to
            target_dtype: Optional dtype to convert weights to

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        if not self.paths.exists():
            self.logger.info(
                f"No checkpoint found at {self.paths.model}, starting fresh"
            )
            return False

        try:
            self.logger.info(f"Loading model weights from {self.paths.model}")
            state_dict = load_safetensors(str(self.paths.model), device="cpu")

            # Clean up DDP prefix if present
            state_dict = _clean_fsdp_state_dict_keys(state_dict)

            # Move to correct device and dtype
            if target_dtype is not None:
                state_dict = {
                    k: v.to(device=device, dtype=target_dtype)
                    for k, v in state_dict.items()
                }
            else:
                state_dict = {k: v.to(device=device) for k, v in state_dict.items()}

            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                self.logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            self.logger.info("Model weights loaded successfully")

            # Also load metadata into bundle if available
            metadata = self._load_metadata()
            if metadata:
                self.bundle.epoch = metadata.epoch
                self.bundle.global_step = metadata.global_step
                self.bundle.tokens_processed = metadata.tokens_processed
                self.bundle.train_loss = metadata.train_loss
                self.bundle.val_loss = metadata.val_loss

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def load_model_fsdp(
        self,
        model: nn.Module,
        device: torch.device,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> bool:
        """
        Load checkpoint into an FSDP-wrapped model.

        Checks for distributed checkpoint format first (faster), then falls
        back to legacy safetensors format.

        Args:
            model: The FSDP-wrapped model to load weights into
            device: The device the model is on
            target_dtype: The dtype to convert weights to (default: bfloat16)

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        # First, check for distributed checkpoint format (new fast format)
        distributed_ckpt_path = self.checkpoint_dir / "distributed_ckpt"
        if distributed_ckpt_path.exists():
            try:
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint.state_dict import (
                    get_state_dict,
                    set_state_dict,
                    StateDictOptions,
                )

                self.logger.info(
                    f"Loading FSDP model from distributed checkpoint: {distributed_ckpt_path}"
                )

                # Get current sharded state dict structure
                model_state, _ = get_state_dict(
                    model,
                    optimizers=None,
                    options=StateDictOptions(
                        full_state_dict=False,
                        cpu_offload=True,
                    ),
                )

                state_dict = {"model": model_state}

                # Load sharded checkpoint
                dcp.load(
                    state_dict,
                    checkpoint_id=str(distributed_ckpt_path),
                )

                # Apply loaded state to model
                set_state_dict(
                    model,
                    optimizers=None,
                    model_state_dict=state_dict["model"],
                    optim_state_dict=None,
                    options=StateDictOptions(
                        full_state_dict=False,
                        cpu_offload=True,
                    ),
                )

                self.logger.info(
                    "FSDP model loaded from distributed checkpoint successfully"
                )

                # Load metadata
                metadata = self._load_metadata()
                if metadata:
                    self.bundle.epoch = metadata.epoch
                    self.bundle.global_step = metadata.global_step
                    self.bundle.tokens_processed = metadata.tokens_processed
                    self.bundle.train_loss = metadata.train_loss
                    self.bundle.val_loss = metadata.val_loss

                return True

            except Exception as e:
                self.logger.warning(
                    f"Failed to load distributed checkpoint: {e}, trying legacy format"
                )

        # Fall back to legacy safetensors format
        if not self.paths.exists():
            self.logger.info(
                f"No checkpoint found at {self.paths.model}, starting fresh"
            )
            return False

        try:
            self.logger.info(
                f"Loading model weights from {self.paths.model} (FSDP legacy mode)"
            )
            # Load full state dict on CPU first (memory efficient)
            state_dict = load_safetensors(str(self.paths.model), device="cpu")

            # Clean up any DDP/FSDP prefixes from previous saves
            cleaned_state_dict = _clean_fsdp_state_dict_keys(state_dict)

            # Try using FSDP's state dict helpers
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                with FSDP.state_dict_type(
                    model,
                    state_dict_type=FSDP.StateDictType.FULL_STATE_DICT,
                ):
                    # Move tensors to correct dtype
                    cleaned_state_dict = {
                        k: v.to(dtype=target_dtype)
                        for k, v in cleaned_state_dict.items()
                    }
                    incompatible = model.load_state_dict(
                        cleaned_state_dict, strict=False
                    )
            except (ImportError, AttributeError, RuntimeError) as e:
                # Fallback for FSDP2 (fully_shard) which may not have state_dict_type
                self.logger.debug(
                    f"FSDP1 state_dict_type not available: {e}, using direct load"
                )
                cleaned_state_dict = {
                    k: v.to(dtype=target_dtype) for k, v in cleaned_state_dict.items()
                }
                incompatible = model.load_state_dict(cleaned_state_dict, strict=False)

            if incompatible.missing_keys:
                self.logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                self.logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            self.logger.info("Model weights loaded successfully (FSDP legacy mode)")

            # Also load metadata into bundle if available
            metadata = self._load_metadata()
            if metadata:
                self.bundle.epoch = metadata.epoch
                self.bundle.global_step = metadata.global_step
                self.bundle.tokens_processed = metadata.tokens_processed
                self.bundle.train_loss = metadata.train_loss
                self.bundle.val_loss = metadata.val_loss

            return True

        except Exception as e:
            self.logger.warning(
                f"FSDP checkpoint loading failed: {e}, continuing with fresh weights"
            )
            return False

    def exists(self) -> bool:
        """Check if a valid checkpoint exists."""
        return self.paths.exists()

    def get_metadata(self) -> Optional[CheckpointMetadata]:
        """Get metadata from existing checkpoint without loading weights."""
        if not self.paths.metadata.exists():
            return None
        return self._load_metadata()

    # =========================================================================
    # Internal Loading Helpers
    # =========================================================================

    def _load_optimizer_state(self) -> None:
        """Load optimizer state from checkpoint into bundle's optimizer."""
        if not self.paths.optimizer.exists():
            self.logger.debug("No optimizer state found in checkpoint")
            return

        try:
            optimizer_state = torch.load(self.paths.optimizer, map_location="cpu")
            # Move tensors to device
            if self.bundle.device is not None:
                for state in optimizer_state.get("state", {}).values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device=self.bundle.device)
            self.bundle.optimizer.load_state_dict(optimizer_state)
            self.logger.info("Optimizer state loaded successfully")
        except Exception as e:
            # Optimizer state loading can fail due to parameter ID mismatch
            # In these cases, we continue with fresh optimizer state
            self.logger.warning(
                f"Failed to load optimizer state (will use fresh optimizer): {e}"
            )

    def _load_loss_fn_state(
        self,
    ) -> None:
        """Load loss function state if available."""
        if self.paths.loss_fn.exists():
            loss_fn_state = torch.load(
                self.paths.loss_fn,
                map_location=torch.device("cpu"),
            )
            self.bundle.loss_fn.load_state_dict(loss_fn_state)
            self.logger.debug("Loss function state loaded successfully")
        else:
            self.logger.warning(
                "Loss function provided but no state in checkpoint. "
                "Using random initialization!"
            )

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    def _save_metadata(self, metadata: CheckpointMetadata) -> None:
        """Save metadata to JSON file."""
        data = metadata.to_dict()
        self._atomic_save_json(data, self.paths.metadata)

    def _load_metadata(self) -> CheckpointMetadata:
        """Load metadata from JSON file."""
        if not self.paths.metadata.exists():
            return CheckpointMetadata()
        with open(self.paths.metadata, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointMetadata.from_dict(data)

    # =========================================================================
    # File Operations
    # =========================================================================

    def _ensure_directory(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def _sync_device(self, device: torch.device) -> None:
        """Sync device before save operations (TPU/XLA or CUDA)."""
        if HAS_XLA and device.type == "xla":
            xm.mark_step()
        elif device.type == "cuda":
            # Synchronize CUDA to ensure all operations are complete
            torch.cuda.synchronize(device)

    def _atomic_save_safetensors(
        self,
        state_dict: dict[str, torch.Tensor],
        filepath: Path,
    ) -> None:
        """Atomically save state dict as safetensors."""
        temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"
        try:
            flat_state_dict = _flatten_state_dict(state_dict)
            # Convert all tensors to clean CPU tensors with valid storage.
            # Handles DTensor from FSDP2, XLA tensors, shared storage, etc.
            clean_state_dict = {
                k: _to_clean_cpu_tensor(v) for k, v in flat_state_dict.items()
            }
            save_safetensors(clean_state_dict, str(temp_filepath))
            shutil.move(str(temp_filepath), str(filepath))
            self.logger.debug(f"Saved safetensors to {filepath}")
        except Exception:
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise

    def _atomic_save_safetensors_direct(
        self,
        state_dict: dict[str, torch.Tensor],
        filepath: Path,
    ) -> None:
        """Atomically save already-converted CPU tensors as safetensors."""
        temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"
        try:
            flat_state_dict = _flatten_state_dict(state_dict)
            save_safetensors(flat_state_dict, str(temp_filepath))
            shutil.move(str(temp_filepath), str(filepath))
            self.logger.debug(f"Saved safetensors to {filepath}")
        except Exception:
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise

    def _atomic_save_pt(self, state_dict: dict[str, Any], filepath: Path) -> None:
        """Atomically save state dict as PyTorch .pt file."""
        temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"
        try:
            torch.save(state_dict, temp_filepath)
            shutil.move(str(temp_filepath), str(filepath))
            self.logger.debug(f"Saved .pt to {filepath}")
        except Exception:
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise

    def _atomic_save_json(self, data: dict[str, Any], filepath: Path) -> None:
        """Atomically save data as JSON file."""
        temp_filepath = filepath.parent / f".tmp_{filepath.name}.{os.getpid()}"
        try:
            serializable_data = _make_json_serializable(data)
            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            shutil.move(str(temp_filepath), str(filepath))
            self.logger.debug(f"Saved JSON to {filepath}")
        except Exception:
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise


# =============================================================================
# Helper Functions
# =============================================================================


def _clean_state_dict_keys(
    state_dict: dict[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """
    Clean state dict keys to match target model wrapper type.

    Handles conversion between:
    - DDP/DataParallel checkpoint -> regular model (remove 'module.' prefix)
    - Regular checkpoint -> DDP/DataParallel model (add 'module.' prefix)
    """
    logger = get_logger()

    checkpoint_has_prefix = any(k.startswith("module.") for k in state_dict.keys())
    model_is_wrapped = isinstance(
        model,
        (nn.DataParallel, nn.parallel.DistributedDataParallel),
    )

    if checkpoint_has_prefix and not model_is_wrapped:
        logger.info("Removing 'module.' prefix from checkpoint keys")
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not checkpoint_has_prefix and model_is_wrapped:
        logger.info("Adding 'module.' prefix to checkpoint keys")
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _clean_fsdp_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Clean state dict keys for FSDP model loading.

    Removes common prefixes added by DDP, FSDP, or compiled models.
    """
    cleaned = {}
    prefixes_to_remove = ["module.", "_orig_mod.", "_fsdp_wrapped_module."]

    for key, value in state_dict.items():
        clean_key = key
        for prefix in prefixes_to_remove:
            if clean_key.startswith(prefix):
                clean_key = clean_key.replace(prefix, "", 1)
        cleaned[clean_key] = value

    return cleaned


def _flatten_state_dict(
    state_dict: dict[str, Any],
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Flatten nested state dict to safetensors-compatible format."""
    flat = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, torch.Tensor):
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_state_dict(value, f"{full_key}."))
    return flat


def _to_clean_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to a clean CPU tensor with valid storage.

    This is needed for safetensors compatibility, especially with:
    - XLA/TPU tensors where storage may be invalid after lazy execution
    - CUDA tensors with shared storage (weight tying, views)
    - Tensor subclasses (DTensor from FSDP2, etc.)

    Always creates a fresh regular tensor to guarantee valid storage.
    """
    original_dtype = tensor.dtype

    # Handle DTensor from FSDP2 - use full_tensor() to get regular tensor
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()

    original_device = tensor.device

    # Sync CUDA before moving to CPU to ensure all operations are complete
    if original_device.type == "cuda":
        torch.cuda.synchronize(original_device)

    # Detach and move to CPU
    cpu_tensor = tensor.detach().cpu()

    # If still a tensor subclass, try to_local() as fallback
    if type(cpu_tensor) is not torch.Tensor:
        if hasattr(cpu_tensor, "to_local"):
            cpu_tensor = cpu_tensor.to_local()

    # Create fresh tensor via clone to ensure clean storage
    result = cpu_tensor.clone().contiguous()

    # Ensure correct dtype
    if result.dtype != original_dtype:
        result = result.to(original_dtype)

    return result


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
    elif hasattr(obj, "item"):
        return obj.item()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _should_save_on_this_rank() -> bool:
    """Determine if current rank should save checkpoint."""
    return not dist.is_available() or not dist.is_initialized() or is_rank_0()


def get_checkpoint_manager(
    checkpoint_dir: Path | str | None = None,
) -> DattaBotCheckpointManager:
    """
    Get the singleton DattaBotCheckpointManager instance.

    Args:
        checkpoint_dir: Directory for checkpoints (required on first call,
                       optional on subsequent calls)

    Returns:
        The singleton DattaBotCheckpointManager instance

    Usage:
        # First call - configure the manager
        manager = get_checkpoint_manager(checkpoint_dir="/path/to/checkpoints")

        # Configure bundle references
        manager.bundle.unwrapped_model = model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = device

        # Save/load using simple API
        manager.save_agent()
        metadata = manager.load_agent()
    """
    return DattaBotCheckpointManager(checkpoint_dir=checkpoint_dir)
