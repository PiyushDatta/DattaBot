from torch import nn
from omegaconf import DictConfig


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
