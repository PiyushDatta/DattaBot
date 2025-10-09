from math import sqrt
from typing import Optional

from src.agent_config import get_agent_config
from src.logger import get_logger
from src.tokenizer import get_tokenizer
from src.util import get_logging_level_from_config
from torch import arange, backends, cat, dtype, nn, ones, rsqrt, Tensor


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
        self.pos_embedding = nn.Embedding(self.max_tokens, self.d_model)
        self.emb_dropout = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.decoder_stack = TransformerDecoderStack(
            n_layers=self.n_layers,
            d_model=self.d_model,
        )
        self.final_norm = RMSNorm(dim=self.d_model)

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
        # Embeddings + positional encoding + dropout
        # gpt-style embedding scaling
        output = self.token_embedding(input_ids) * sqrt(self.d_model)
        output = output + self.pos_embedding(pos_ids)
        output = self.emb_dropout(output)
        # apply causal mask inside
        output = self.decoder_stack(output, attention_pad_mask)
        # final norm
        logits = self.final_norm(output)
        # [batch, seq_len] -> [batch, seq_len, d_model]
        return logits


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
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
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
            output = layer(tgt_input=output, attention_mask=attention_mask)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embedded_dim_size: int,
    ) -> None:
        super().__init__()
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

    def forward(
        self, tgt_input: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            tgt_input: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len, seq_len] causal mask (optional)
        """
        # 1. Masked self-attention + residual connection.
        output = self.multi_head_attn(
            input_query=tgt_input,
            input_key=tgt_input,
            input_value=tgt_input,
            mask=attention_mask,
        )
        output = self.multi_head_attn_dropout(output)
        output = tgt_input + output
        output = self.multi_head_attn_norm(output)
        # 2. Feed-forward network + residual connection
        output = output + self.position_wise_ffn(output)
        output = self.position_wise_ffn_norm(output)
        return output


class TransformerPositionWiseFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.n_heads = self.config.neural_net.n_heads
        self.d_model = self.config.neural_net.model_dimensions
        self.linear_one = nn.Linear(self.d_model, self.config.neural_net.hidden_layers)
        self.linear_two = nn.Linear(self.config.neural_net.hidden_layers, self.d_model)
        self.gelu_layer = nn.GELU()
        self.dropout_one = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.dropout_two = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(self, src_input) -> Tensor:
        output = self.linear_one(src_input)
        output = self.gelu_layer(output)
        output = self.dropout_one(output)
        output = self.linear_two(output)
        return self.dropout_two(output)


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.n_heads = self.config.neural_net.n_heads
        self.attention_output_size = embedded_dim_size // self.n_heads
        self.attention_layers = nn.ModuleList(
            [
                TransformerScaledDotProductAttention(
                    output_size=self.attention_output_size,
                    embedded_dim_size=embedded_dim_size,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.concat_layer = nn.Linear(embedded_dim_size, embedded_dim_size)

    def forward(
        self, input_query: Tensor, input_key: Tensor, input_value: Tensor, mask: Tensor
    ) -> Tensor:
        attention_score = cat(
            [
                layer(query=input_query, key=input_key, value=input_value, mask=mask)
                for layer in self.attention_layers
            ],
            dim=-1,
        )
        return self.concat_layer(attention_score)


class TransformerScaledDotProductAttention(nn.Module):
    def __init__(self, output_size, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.output_size = output_size
        self.pad_id = get_tokenizer().pad_token_id
        self.query_layer = nn.Linear(
            in_features=embedded_dim_size, out_features=output_size
        )
        self.key_layer = nn.Linear(
            in_features=embedded_dim_size, out_features=output_size
        )
        self.value_layer = nn.Linear(
            in_features=embedded_dim_size, out_features=output_size
        )
        self.dropout = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        # TODO(PiyushDatta): Figure out what to do with the mask, we are padding the sequences.
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [batch_size, seq_len, d_model]
        _, tgt_len, _ = query.size()
        sequence_len = key.size()[1]
        assert sequence_len == tgt_len, (
            f"Sequence length must equal target length! "
            f"Got {sequence_len}, expected {tgt_len}"
        )
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)
        # PyTorch SDPA expects [batch, n_heads, seq, head_dim]
        # Right now, each "attention head" is wrapped in TransformerMultiHeadAttention,
        # so q/k/v are already split per head. Shapes are [batch, seq_len, head_dim].
        # [batch, 1, seq_len, head_dim]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        # Use SDPA (dispatches to FlashAttention if available)
        with backends.cuda.sdp_kernel(
            enable_flash=backends.cuda.flash_sdp_enabled(),
            enable_math=False,
            enable_mem_efficient=False,
        ):
            out = nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        # [batch, 1, seq_len, head_dim] -> [batch, seq_len, head_dim]
        return out.squeeze(1)


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
