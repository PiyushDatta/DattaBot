import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        Root Mean Square Layer Normalization.
        https://arxiv.org/abs/1910.07467

        Use this instead of regular LayerNorm if we do not want to center around the mean and want no bias.
        Less memory per normalization layer and faster than LayerNorm.

        Args:
            eps:  epsilon value
            dim:  input dimension
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Optimized RMSNorm from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L35
        # Use rsqrt instead of sqrt to avoid division, and do multiplication instead.
        # Multiplication is faster on GPU, than division.
        # Also in nvidia GPUs, rsqrt is faster than sqrt in practice.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
