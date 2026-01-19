from math import sqrt
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.model.moe import MixtureOfExperts
from src.model.rms_norm import RMSNorm
from src.tokenizer import get_tokenizer
from src.util import get_logging_level_from_config
from torch import (
    arange,
    backends,
    cat,
    device as torch_device,
    dtype,
    einsum,
    empty,
    nn,
    ones,
    outer,
    randn_like,
    rsqrt,
    tensor,
    Tensor,
    topk,
    utils as torch_utils,
    zeros,
)
from torch.nn.attention import sdpa_kernel, SDPBackend


def _is_gradient_checkpointing_enabled(device: torch_device) -> bool:
    config = get_agent_config()
    return (
        config.neural_net.gradient_checkpointing
        and device.type != "xla"
        and not device.type.startswith("xla")
    )


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, device: torch_device, dtype: dtype = "float32") -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.device = device
        self.dtype = dtype
        self.vocab_size = get_tokenizer().vocab_size
        assert (
            self.vocab_size is not None and self.vocab_size > 0
        ), f"Invalid vocab size: {self.vocab_size}"
        self.n_layers = self.config.neural_net.n_layers
        self.n_heads = self.config.neural_net.n_heads
        self.d_model = self.config.neural_net.model_dimensions
        # Max tokens for response.
        self.max_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Max tokens: {self.max_tokens}")
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.emb_dropout = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.decoder_stack = TransformerDecoderStack(
            n_layers=self.n_layers, d_model=self.d_model, device=self.device
        )
        self.final_norm = RMSNorm(dim=self.d_model)
        self.gradient_checkpointing = _is_gradient_checkpointing_enabled(
            device=self.device
        )
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing to reduce memory usage.
        Trades compute for memory savings.
        """
        for layer in self.decoder_stack.layers:
            layer.gradient_checkpointing = True
        self.logger.info(
            "Gradient checkpointing enabled for all decoder layers.", all_ranks=True
        )

    def init_weights(self) -> None:
        # TODO(PiyushDatta): Add initialization of weights.
        pass

    @property
    def layers(self):
        return self.decoder_stack.layers

    def forward(
        self, input_ids: Tensor, attention_pad_mask: Optional[Tensor] = None
    ) -> Tensor:
        self.logger.debug(
            f"Model received the following tensor inputs:\n"
            f"input_ids: {input_ids}, with shape: {input_ids.shape}\n"
            f"attention_pad_mask: {attention_pad_mask if attention_pad_mask is not None else 'None'}, "
            f"with shape: {attention_pad_mask.shape if attention_pad_mask is not None else 'N/A'}"
        )
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.size()
        pos_ids = (
            arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        # Embeddings + RoPe relative positional encoding + dropout
        # gpt-style embedding scaling
        output = self.token_embedding(input_ids) * sqrt(self.d_model)
        output = self.emb_dropout(output)
        # apply causal mask inside
        output = self.decoder_stack(output, attention_pad_mask, pos_ids)
        # final norm
        logits = self.final_norm(output)
        # [batch, seq_len] -> [batch, seq_len, d_model]
        return logits

    def get_load_balancing_loss(self) -> Tensor:
        """
        Collect load balancing losses from all MoE layers.
        Returns a scalar tensor (0 if no MoE is used).
        """
        total_loss = tensor(0.0, device=self.device)
        for layer in self.decoder_stack.layers:
            layer_loss = layer.get_load_balancing_loss()
            if layer_loss is not None:
                total_loss = total_loss + layer_loss
        return total_loss


class TransformerDecoderStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, device: torch_device) -> None:
        super().__init__()
        # Multiple attention layers.
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embedded_dim_size=d_model, device=device)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            input_ids: [batch, seq_len, d_model] (token + pos embeddings applied)
            attention_mask: optional causal/padding mask [batch, seq_len, seq_len]

        Returns:
            [batch, seq_len, d_model]
        """
        output = input_ids
        for layer in self.layers:
            output = layer(
                tgt_input=output,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embedded_dim_size: int,
        device: torch_device,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        # Self attention.
        self.multi_head_attn = TransformerMultiHeadAttention(
            embedded_dim_size=embedded_dim_size, device=device
        )
        self.multi_head_attn_norm = RMSNorm(dim=embedded_dim_size)
        self.multi_head_attn_dropout = nn.Dropout(
            p=get_agent_config().neural_net.zeroed_drop_probability
        )
        # Feed forward.
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.position_wise_ffn_norm = RMSNorm(dim=embedded_dim_size)
        self.gradient_checkpointing = _is_gradient_checkpointing_enabled(device=device)

    def forward(
        self,
        tgt_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Make sure we normalize before the feed forward layers.

        Args:
            tgt_input: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len, seq_len] causal mask (optional)
        """
        # 1. Masked self-attention + residual connection.
        norm_output = self.multi_head_attn_norm(tgt_input)
        if self.gradient_checkpointing and self.training:
            attn_output = torch_utils.checkpoint.checkpoint(
                self.multi_head_attn,
                norm_output,
                norm_output,
                norm_output,
                attention_mask,
                position_ids,
                use_reentrant=False,
            )
        else:
            attn_output = self.multi_head_attn(
                input_query=norm_output,
                input_key=norm_output,
                input_value=norm_output,
                mask=attention_mask,
                position_ids=position_ids,
            )
        attn_output = self.multi_head_attn_dropout(attn_output)
        output = tgt_input + attn_output
        # 2. Feed-forward network + residual connection
        ffn_norm_output = self.position_wise_ffn_norm(output)
        if self.gradient_checkpointing and self.training:
            ffn_output = torch_utils.checkpoint.checkpoint(
                self.position_wise_ffn, ffn_norm_output, use_reentrant=False
            )
        else:
            ffn_output = self.position_wise_ffn(ffn_norm_output)
        return output + ffn_output

    def get_load_balancing_loss(self) -> Optional[Tensor]:
        """Get load balancing loss from FFN if it uses MoE."""
        return self.position_wise_ffn.get_load_balancing_loss()


class TransformerPositionWiseFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.n_heads = self.config.neural_net.n_heads
        self.d_model = self.config.neural_net.model_dimensions
        self.dropout_prob = self.config.neural_net.zeroed_drop_probability
        self.use_moe = self.config.neural_net.use_moe

        if self.use_moe:
            self.moe = MixtureOfExperts()
        else:
            self.linear_one = nn.Linear(
                self.d_model, self.config.neural_net.hidden_layers
            )
            self.linear_two = nn.Linear(
                self.config.neural_net.hidden_layers, self.d_model
            )
            self.gelu_layer = nn.GELU()
            self.dropout_one = nn.Dropout(p=self.dropout_prob)
            self.dropout_two = nn.Dropout(p=self.dropout_prob)

    def forward(self, src_input) -> Tensor:
        if self.use_moe:
            assert self.moe is not None
            return self.moe(src_input)
        output = self.linear_one(src_input)
        output = self.gelu_layer(output)
        output = self.dropout_one(output)
        output = self.linear_two(output)
        return self.dropout_two(output)

    def get_load_balancing_loss(self) -> Optional[Tensor]:
        """Get the load balancing loss from MoE if applicable."""
        if self.use_moe:
            return self.moe.get_load_balancing_loss()
        return None


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, embedded_dim_size: int, device: torch_device) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.n_heads = self.config.neural_net.n_heads
        self.d_model = embedded_dim_size
        self.head_dim = embedded_dim_size // self.n_heads
        # Single linear layers for all heads (batched)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=self.config.agent.max_response_tokens * 2,
            device=device,
        )
        self.dropout = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(
        self,
        input_query: Tensor,
        input_key: Tensor,
        input_value: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, d_model = input_query.shape
        q = self.q_proj(input_query)
        k = self.k_proj(input_key)
        v = self.v_proj(input_value)
        if q.dtype != v.dtype or k.dtype != v.dtype:
            # Safety check, all should have the same dtype.
            # This shouldn't happen if model is properly initialized.
            self.logger.warning(
                f"Dtype mismatch in attention: q={q.dtype}, k={k.dtype}, v={v.dtype}. "
                f"This indicates improper model initialization."
            )
            target_dtype = v.dtype
            q = q.to(target_dtype)
            k = k.to(target_dtype)
        # Reshape to split heads: [batch, seq, d_model] -> [batch, seq, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # Transpose for attention: [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Rope to all head at once.
        q, k = self.rope(q, k, position_ids)
        # Flash Attention (batched across all heads).
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        # Merge heads.
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(out)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    https://arxiv.org/abs/2104.09864

    Allows model to understand relative positioning instead of absolute positioning.
    Also enables better long range dependencies.
    Allows model to evaluate on longer sequences.
    Slightly better model performance, training speed and memory efficiency,
    than absolute embedding positioning.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Compute inverse frequencies.
        inv_freq = 1.0 / (base ** (arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cos/sin cache.
        self._build_cache(max_seq_len, device)

    def _build_cache(self, max_seq_len: int, device: str):
        """Precompute rotation matrices for all positions."""
        t = arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = outer(t, self.inv_freq)
        emb = cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """Apply rotary embeddings to input tensor.
        Args:
            x: [batch, n_heads, seq, head_dim] or [batch, seq, head_dim]
            position_ids: [batch, seq]
        Returns:
            Rotated tensor with same shape as x
        """
        # Index cos/sin cache: [batch, seq, head_dim].
        cos = self.cos_cached[position_ids].to(x.device)
        sin = self.sin_cached[position_ids].to(x.device)
        # Unsqueeze for multi-head attention if needed.
        if x.dim() == 4:
            # x: [batch, n_heads, seq, head_dim]
            # cos/sin: [batch, seq, head_dim]
            # Unsqueeze to [batch, 1, seq, head_dim] for broadcasting
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(
        self, q: Tensor, k: Tensor, position_ids: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.
        Args:
            q: [batch, n_heads, seq, head_dim]
            k: [batch, n_heads, seq, head_dim]
            position_ids: [batch, seq] or None
        Returns:
            (q_rotated, k_rotated)
        """
        batch_size, _, seq_len, _ = q.shape
        # Generate position IDs if not provided.
        if position_ids is None:
            position_ids = (
                arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
            )
        # Rebuild cache if needed.
        if seq_len > self.max_seq_len or self.cos_cached.device != q.device:
            self._build_cache(max(seq_len, self.max_seq_len), q.device)
        # Apply rotation.
        q_rotated = self.apply_rotary_pos_emb(q, position_ids)
        k_rotated = self.apply_rotary_pos_emb(k, position_ids)
        return q_rotated, k_rotated
