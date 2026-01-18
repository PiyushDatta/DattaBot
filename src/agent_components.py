from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from src.logger import get_logger
from src.tokenizer import DattaBotTokenizer
from src.util import get_logging_level_from_config
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR

# TPU support
try:
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


@dataclass
class DattaBotAgentComponents:
    """Container for agent components."""

    loss_fn: nn.Module
    lr: float
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    scaler: Optional[torch.amp.GradScaler]
    max_response_tokens: int
    use_moe: bool
    moe_weight: float
    weights_dir: Path
    data_dir: Path
    plot_dir: Path


class DattaBotAgentComponentFactory:
    """
    Factory class for creating agent components.

    Usage:
        factory = DattaBotAgentComponentFactory(config=config, device=device)
        agent_components = factory.create()
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: DattaBotTokenizer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        device_info: dict[str, any],
        local_rank: int,
        tensor_dtype: torch.dtype,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.device_info = device_info
        self.tensor_dtype = tensor_dtype
        self.autocast_dtype = autocast_dtype
        self.local_rank = local_rank
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )

    def create(self) -> DattaBotAgentComponents:
        """Create agent components."""
        return DattaBotAgentComponents(
            loss_fn=self._create_loss_fn(),
            lr=self.config.agent.lr,
            lr_scheduler=self._create_scheduler(),
            scaler=self._create_scaler(),
            max_response_tokens=self.config.agent.max_response_tokens,
            use_moe=self.config.neural_net.use_moe,
            moe_weight=self.config.neural_net.moe_load_balance_weight,
            weights_dir=self._create_path_dir(self.config.agent.weights_file_name),
            data_dir=self._create_path_dir(
                self.config.agent.data_directory, create_if_not_exist=True
            ),
            plot_dir=self._create_path_dir(
                self.config.agent.plot_directory, create_if_not_exist=True
            ),
        )

    def _create_loss_fn(self) -> nn.Module:
        """Create the loss function based on config."""
        # Initialize AdaptiveLogSoftmaxWithLoss.
        # Vocab size is huge and increases the memory during forward pass by a
        # lot, to reduce the memory overhead we use AdaptiveLogSoftmaxWithLoss
        # rather than standard softmax. Tradeoff between accuracy and memory.
        # https://arxiv.org/abs/1609.04309
        return nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.config.neural_net.model_dimensions,
            n_classes=self.tokenizer.vocab_size,
            cutoffs=[10_000, 50_000, self.tokenizer.vocab_size - 1],
            div_value=4.0,
            head_bias=False,
        ).to(self.device, dtype=self.autocast_dtype)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Only training epochs, do not include validation epochs.
        return TorchOneCycleLR(
            self.optimizer,
            max_lr=self.config.agent.lr,
            epochs=self.config.agent.max_training_num_epochs,
            steps_per_epoch=self.config.agent.num_batches_train_every_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
        )

    def _create_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Create gradient scaler if we can."""
        scaler = None
        if not HAS_XLA and self.autocast_dtype != torch.bfloat16:
            # Initialize AMP scaler.
            # Reduce memory consumption and improve training speed.
            # https://arxiv.org/abs/1710.03740
            scaler = torch.amp.GradScaler(device=self.device)
        return scaler

    def _create_path_dir(self, path: str, create_if_not_exist: bool = False) -> Path:
        """Create Path object from given directory."""
        path_dir = Path(path)
        if not path_dir.exists():
            if create_if_not_exist:
                path_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Path not found: {path_dir}")
        return path_dir
