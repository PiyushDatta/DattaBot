from math import sqrt
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from src.agent_config import get_agent_config
from src.logger import get_logger
from src.tokenizer import get_tokenizer
from src.util import get_logging_level_from_config
from torch import (
    arange,
    backends,
    cat,
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


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, device: str = "cpu", dtype: dtype = "float32") -> None:
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
            n_layers=self.n_layers,
            d_model=self.d_model,
        )
        self.final_norm = RMSNorm(dim=self.d_model)
        self.gradient_checkpointing = self.config.neural_net.gradient_checkpointing
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
    def __init__(
        self,
        n_layers: int,
        d_model: int,
    ) -> None:
        super().__init__()
        # Multiple attention layers.
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embedded_dim_size=d_model)
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
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        # Self attention.
        self.multi_head_attn = TransformerMultiHeadAttention(
            embedded_dim_size=embedded_dim_size
        )
        self.multi_head_attn_norm = RMSNorm(dim=embedded_dim_size)
        self.multi_head_attn_dropout = nn.Dropout(
            p=get_agent_config().neural_net.zeroed_drop_probability
        )
        # Feed forward.
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.position_wise_ffn_norm = RMSNorm(dim=embedded_dim_size)
        self.gradient_checkpointing = self.config.neural_net.gradient_checkpointing

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
    def __init__(self, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
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
            device=self.config.env.device,
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
        with backends.cuda.sdp_kernel(
            enable_flash=backends.cuda.flash_sdp_enabled(),
            enable_math=False,
            enable_mem_efficient=False,
        ):
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
        self.weight = nn.Parameter(ones(dim))

    def _norm(self, x):
        # Optimized RMSNorm from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L35
        # Use rsqrt instead of sqrt to avoid division, and do multiplication instead.
        # Multiplication is faster on GPU, than division.
        # Also in nvidia GPUs, rsqrt is faster than sqrt in practice.
        return x * rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture-of-Experts Layer with Top-K Routing.
    Papers:
        1. https://arxiv.org/abs/1701.06538
        2. https://arxiv.org/abs/2101.03961

    Mathematical formulation:
        y = Σ(i=1 to num_experts_per_token) G(x)_i * E_i(x)

    Where:
        - G(x) is the gating/routing function (produces weights)
        - E_i(x) is the i-th expert network
        - num_experts_per_token is the number of experts activated per token (top-k routing)
        - y is the final output (weighted sum of expert outputs)
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = get_agent_config()
        # Model dimensions
        self.d_model = self.config.neural_net.model_dimensions
        self.hidden_dim = self.config.neural_net.hidden_layers
        self.dropout_prob = self.config.neural_net.zeroed_drop_probability
        # MoE hyperparameters
        self.num_experts = self.config.neural_net.num_experts
        self.num_experts_per_token = self.config.neural_net.num_experts_per_token
        self.norm = RMSNorm(dim=self.d_model)
        # Gating network G(x): maps input to expert scores
        # G: R^d_model -> R^num_experts
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        # Expert networks E_i(x) stored as batched parameters
        # All experts share the same architecture but have different weights
        # Each expert is a 2-layer MLP: E_i(x) = W2_i * σ(W1_i * x + b1_i) + b2_i
        # First layer weights: W1 ∈ R^(num_experts × hidden_dim × d_model)
        self.expert_fc1_weight = nn.Parameter(
            empty(self.num_experts, self.hidden_dim, self.d_model)
        )
        # First fully connected (input) layer biases for all experts: [num_experts, hidden_dim]
        self.expert_fc1_bias = nn.Parameter(empty(self.num_experts, self.hidden_dim))
        # Second layer weights: W2 ∈ R^(num_experts × d_model × hidden_dim)
        self.expert_fc2_weight = nn.Parameter(
            empty(self.num_experts, self.d_model, self.hidden_dim)
        )
        # Second fully connected (output) layer biases for all experts: [num_experts, d_model]
        self.expert_fc2_bias = nn.Parameter(empty(self.num_experts, self.d_model))
        # Activation and regularization
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.dropout2 = nn.Dropout(p=self.dropout_prob)
        # Initialize expert parameters
        self._init_expert_weights()
        # Auxiliary loss for load balancing (prevents expert collapse)
        self.aux_loss = None

    def _init_expert_weights(self) -> None:
        """
        Initialize expert weights using Xavier uniform initialization.
        Ensures proper gradient flow at start of training.
        """
        nn.init.xavier_uniform_(self.expert_fc1_weight)
        nn.init.zeros_(self.expert_fc1_bias)
        nn.init.xavier_uniform_(self.expert_fc2_weight)
        nn.init.zeros_(self.expert_fc2_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Sparse MoE forward pass with top-k routing.
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        Paper notation:
            output = Σ(i=1 to num_experts_per_token) G(x)_i * E_i(x)
        """
        batch_size, seq_len, _ = x.shape
        # Apply pre-normalization
        t = self.norm(x)
        # Flatten batch and sequence dimensions for efficient processing
        # tokens: [num_tokens, d_model] where num_tokens = batch_size * seq_len
        tokens = t.view(-1, self.d_model)
        # ========================================
        # Step 1: Gating - Compute G(x)
        # ========================================
        # Compute routing logits for each token
        # gate_logits: [num_tokens, num_experts]
        gate_logits = self.gate(tokens)
        # Add noise during training for exploration (jitter)
        if self.training:
            noise = randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        # Select top-k experts per token
        # expert/gate_scores: [num_tokens, num_experts_per_token] - unnormalized scores
        # expert_indices: [num_tokens, num_experts_per_token] - which experts to use
        gate_scores, expert_indices = topk(
            gate_logits, k=self.num_experts_per_token, dim=-1, sorted=False
        )
        # Normalize gate scores to get routing weights (softmax over top-k)
        # routing_weights: [num_tokens, num_experts_per_token] - G(x)_i in paper
        routing_weights = F.softmax(gate_scores, dim=-1)
        # =========================================
        # Step 2: Load Balancing Loss (if training)
        # =========================================
        if self.training:
            self.aux_loss = self._compute_load_balancing_loss(gate_logits)
        else:
            self.aux_loss = None
        # ========================================
        # Step 3: Expert Computation - E_i(x)
        # ========================================
        # Gather weights for selected experts
        # MLP #1: [batch*seq, experts_per_token, hidden_dim, d_model]
        expert_fc1_weight = self.expert_fc1_weight[expert_indices, ...]
        expert_fc1_bias = self.expert_fc1_bias[expert_indices, ...]
        # Apply first layer: einsum "beck,bk->bec"
        # b: batch*seq, e: experts_per_token, c: hidden_dim*2, k: d_model
        hidden = einsum("beck,bk->bec", expert_fc1_weight, tokens) + expert_fc1_bias
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)

        # Gather second layer weights
        # W2_selected: [num_tokens, num_experts_per_token, d_model, hidden_dim]
        # b2_selected: [num_tokens, num_experts_per_token, d_model]
        expert_fc2_weight = self.expert_fc2_weight[expert_indices, ...]
        expert_fc2_bias = self.expert_fc2_bias[expert_indices, ...]
        # Apply second layer: einsum "beck,bek->bec"
        # b: batch*seq, e: experts_per_token, c: d_model, k: hidden_dim
        expert_outputs = (
            einsum("beck,bek->bec", expert_fc2_weight, hidden) + expert_fc2_bias
        )
        expert_outputs = self.dropout2(expert_outputs)
        # ========================================
        # Step 4: Weighted Aggregation
        # ========================================
        # Combine expert outputs using routing weights: einsum "bec,be->bc"
        # b: batch*seq, e: experts_per_token, c: d_model
        output = einsum("bec,be->bc", expert_outputs, routing_weights)
        # Reshape back to [batch_size, seq_len, d_model]
        output = output.view(batch_size, seq_len, self.d_model)
        return x + output

    def _compute_load_balancing_loss(self, gate_logits: Tensor) -> Tensor:
        """
        Auxiliary loss to encourage balanced expert usage.
        L_aux = alpha * num_experts * Σ(i=1 to num_experts) f_i * P_i

        Where:
            - num_experts: total number of experts
            - f_i: fraction of tokens routed to expert i
            - P_i: mean router probability for expert i
            - alpha: scaling coefficient (typically small, e.g., 0.01)

        This loss prevents "expert collapse" where all tokens route to
        a small subset of experts, leaving others unused.

        Args:
            gate_logits: [num_tokens, num_experts] - raw routing logits

        Returns:
            aux_loss: scalar - auxiliary load balancing loss
        """
        # Convert logits to probabilities: P_i for each expert
        # router_probs: [num_tokens, num_experts]
        router_probs = F.softmax(gate_logits, dim=-1)

        # Mean probability assigned to each expert (across all tokens).
        # Convert logits to probabilities.
        mean_prob_per_expert = router_probs.mean(dim=0)
        # Mean probability assigned to each expert
        # Fraction of tokens that chose each expert as their top choice.
        top_expert_indices = router_probs.argmax(dim=-1)
        expert_mask = F.one_hot(
            top_expert_indices, num_classes=self.num_experts
        ).float()
        fraction_tokens_per_expert = expert_mask.mean(dim=0)
        # Load balancing loss: num_experts * Σ(f_i * P_i)
        # Encourages f_i ≈ P_i ≈ 1/num_experts (uniform distribution)
        aux_loss = (
            self.num_experts * (fraction_tokens_per_expert * mean_prob_per_expert).sum()
        )
        return aux_loss

    def get_load_balancing_loss(self) -> Optional[Tensor]:
        """
        Returns the auxiliary load balancing loss.
        This should be added to the main loss during training:
            total_loss = main_loss + alpha * aux_loss

        Where alpha is typically 0.01
        """
        return self.aux_loss
