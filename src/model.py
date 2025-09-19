from math import sqrt
from typing import Optional

from numpy import sqrt as np_sqrt
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.tokenizer import get_tokenizer
from torch import arange, bmm, cat, nn, Tensor


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.device = device
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
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        self.logger.debug(
            f"Model received the following tensor inputs:\n"
            f"input_ids: {input_ids}, with shape: {input_ids.shape}\n"
            f"attention_mask: {attention_mask if attention_mask is not None else 'None'}, "
            f"with shape: {attention_mask.shape if attention_mask is not None else 'N/A'}"
        )
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.size()
        pos_ids = (
            arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        # gpt-style embedding scaling
        output = self.token_embedding(input_ids) * sqrt(self.d_model)
        output = output + self.pos_embedding(pos_ids)
        output = self.emb_dropout(output)
        # apply causal mask inside
        output = self.decoder_stack(output, attention_mask)
        # final norm + projection
        output = self.final_norm(output)
        logits = self.output_projection(output)
        # [batch, seq_len, vocab_size]
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
        self.multi_head_attn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)
        self.multi_head_attn_dropout = nn.Dropout(
            p=get_agent_config().neural_net.zeroed_drop_probability
        )
        # Feed forward.
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.position_wise_ffn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)

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
        # TODO(PiyushDatta): Test with relu vs gelu.
        # self.relu_layer = nn.ReLU()
        self.gelu_layer = nn.GELU()
        self.dropout_one = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.dropout_two = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(self, src_input) -> Tensor:
        output = self.linear_one(src_input)
        # TODO(PiyushDatta): Test with relu vs gelu.
        # output = self.relu_layer(output)
        output = self.gelu_layer(output)
        output = self.dropout_one(output)
        output = self.linear_two(output)
        return self.dropout_two(output)


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        # mask = batch size * sequence length
        mask: Tensor = None,
    ) -> Tensor:
        # Tensor input is 3D tensor.
        # [batch_size, sequence_size, embedding_table_size]
        batch_size, tgt_len, _ = query.size()
        sequence_len = key.size()[1]
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)
        dim_k = key.size(-1)
        # Transpose keys. K^T.
        key = key.transpose(1, 2)
        # Get dot product of queries with all keys.
        # Then divide by sqrt of model_dimensions (d_k).
        # This covers the first matmul and scale blocks inside ScaledDotProductAttention.
        dot_product_score = bmm(input=query, mat2=key) / np_sqrt(dim_k)
        # Apply masking (optional).
        if mask is not None:
            # mask = batch size * sequence length * sequence length
            expanded_mask = mask[:, None, :].expand(batch_size, tgt_len, sequence_len)
            dot_product_score = dot_product_score.masked_fill(
                expanded_mask == self.pad_id, -float("Inf")
            )
        # Apply softmax.
        attention_score = self.softmax(dot_product_score)
        attention_score = self.dropout(attention_score)
        # Matmul of our values and final attention score.
        return bmm(input=attention_score, mat2=value)
