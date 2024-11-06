from torch import (
    nn,
    cat,
    bmm,
    zeros,
    arange,
    sin,
    cos,
    Tensor,
    set_default_device as torch_set_default_device,
)
from numpy import sqrt as np_sqrt
from transformers import AutoTokenizer
from src.logger import get_logger
from src.agent_config import get_agent_config


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.device = self.config.env.device
        # TODO(PiyushDatta): Once supported, set default dtype as int64. Currently
        #                    only float types are supported.
        torch_set_default_device(self.device)
        # Assumes tokenizer is already setup.
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.n_layers = self.config.neural_net.n_layers
        self.n_heads = self.config.neural_net.n_heads
        self.model_dimensions = self.config.neural_net.model_dimensions
        # Max tokens for response.
        self.response_max_response_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Max tokens: {self.response_max_response_tokens}")
        self.encoder_stack = TransformerEncoderStack(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            model_embedding_dims=self.model_dimensions,
            tokenizer=tokenizer,
        )
        self.decoder_stack = TransformerDecoderStack(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            model_embedding_dims=self.model_dimensions,
            tokenizer=tokenizer,
        )
        # Output projection layer to convert to the words in our training set.
        self.output_projection = nn.Linear(self.model_dimensions, self.vocab_size)

    def forward(
        self, src_input: Tensor, src_mask: Tensor, tgt_input: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        self.logger.debug(
            f"Model received the following tensor inputs:\n{src_input}, with shape: {src_input.shape}\n{tgt_input}, with shape: {tgt_input.shape}"
        )
        if src_mask is not None:
            self.logger.debug(
                f"Model received the following tensor mask: {src_mask}, with shape: {src_mask.shape}"
            )
        if tgt_mask is not None:
            self.logger.debug(
                f"Model received the following tensor mask: {tgt_mask}, with shape: {tgt_mask.shape}"
            )
        encoder_stack_output = self.encoder_stack(
            src_input=src_input,
            src_mask=src_mask,
        )
        # Return logits.
        return self.output_projection(encoder_stack_output)


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        model_embedding_dims: int,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.model_embedding_dims = model_embedding_dims
        # Embedding layer.
        self.embedding_layer = TransformerEmbedding(
            tokenizer=tokenizer, embedded_dim_size=self.model_embedding_dims
        )
        # Positional Embedding.
        self.positional_encoding = TransformerPositionalEncoding(
            embedded_dim_size=self.model_embedding_dims
        )
        # Multiple attention layers.
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    tokenizer=tokenizer, embedded_dim_size=self.model_embedding_dims
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, src_input: Tensor, src_mask: Tensor) -> Tensor:
        output = self.embedding_layer(src_input)
        output = self.positional_encoding(output)
        for layer in self.layers:
            output = layer(src_input=output, src_mask=src_mask)
        return output


class TransformerDecoderStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        model_embedding_dims: int,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.model_embedding_dims = model_embedding_dims
        # Embedding layer.
        self.embedding_layer = TransformerEmbedding(
            tokenizer=tokenizer, embedded_dim_size=self.model_embedding_dims
        )
        # Positional Embedding.
        self.positional_encoding = TransformerPositionalEncoding(
            embedded_dim_size=self.model_embedding_dims
        )
        # Multiple attention layers.
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    tokenizer=tokenizer, embedded_dim_size=self.model_embedding_dims
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, tgt_input: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        output = self.embedding_layer(tgt_input)
        output = self.positional_encoding(output)
        for layer in self.layers:
            output = layer(src_input=output, src_mask=src_mask)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        embedded_dim_size: int,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.tokenizer = tokenizer
        self.multi_head_attn = TransformerMultiHeadAttention(
            tokenizer=self.tokenizer, embedded_dim_size=embedded_dim_size
        )
        self.multi_head_attn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.position_wise_ffn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)

    def forward(self, src_input: Tensor, src_mask: Tensor) -> Tensor:
        output = src_input
        # Query: seeks specific information in the input.
        # Key: responds to these queries.
        # Value: delivers the content we aim to focus on, based on the alignment
        #        between Query and Key.
        #
        # Source: https://awadrahman.medium.com/from-theory-to-code-make-sense-of-transformers-in-machine-learning-51b8b23c34c5
        output = output + self.multi_head_attn(
            input_query=output, input_key=output, input_value=output, mask=src_mask
        )
        output = self.multi_head_attn_norm(output)
        output = output + self.position_wise_ffn(output)
        return self.position_wise_ffn_norm(output)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        embedded_dim_size: int,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.tokenizer = tokenizer
        # Self attention.
        self.multi_head_attn = TransformerMultiHeadAttention(
            tokenizer=self.tokenizer, embedded_dim_size=embedded_dim_size
        )
        self.multi_head_attn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)
        # Cross attention.
        self.cross_attn = TransformerMultiHeadAttention(
            tokenizer=self.tokenizer, embedded_dim_size=embedded_dim_size
        )
        self.cross_attn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)
        # Feed forward.
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.position_wise_ffn_norm = nn.LayerNorm(normalized_shape=embedded_dim_size)

    def forward(
        self,
        tgt_input: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        output = tgt_input
        # Query: seeks specific information in the input.
        # Key: responds to these queries.
        # Value: delivers the content we aim to focus on, based on the alignment
        #        between Query and Key.
        #
        # Source: https://awadrahman.medium.com/from-theory-to-code-make-sense-of-transformers-in-machine-learning-51b8b23c34c5
        output = output + self.multi_head_attn(
            input_query=output, input_key=output, input_value=output, mask=tgt_mask
        )
        output = self.multi_head_attn_norm(output)
        output = output + self.cross_attention(
            input_query=output,
            input_key=encoder_output,
            input_value=encoder_output,
            mask=src_mask,
        )
        output = self.cross_attention_norm(output)
        # Feed forward.
        output = output + self.position_wise_ffn(output)
        return self.position_wise_ffn_norm(output)


class TransformerEmbedding(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.vocab_size = tokenizer.vocab_size
        self.embeddings_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedded_dim_size
        )

    def forward(self, src_input: Tensor) -> Tensor:
        assert (
            src_input.ndim == 2
        ), f"Expected: (batch size, sequence length), got {src_input.shape}"
        return self.embeddings_table(input=src_input)


class TransformerPositionalEncoding(nn.Module):
    def __init__(self, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.max_response_tokens = self.config.agent.max_response_tokens
        self.pos_encoding = zeros(self.max_response_tokens, embedded_dim_size)
        self.pos = arange(0, self.max_response_tokens).unsqueeze(1)
        # 2i
        two_i = arange(0, embedded_dim_size, step=2).float()
        # PE(pos,2i) = sin(pos/10000^(2i/d_model))
        self.pos_encoding[:, 0::2] = sin(
            self.pos / 10_000 ** (two_i / embedded_dim_size)
        )
        # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        self.pos_encoding[:, 1::2] = cos(
            self.pos / 10_000 ** (two_i / embedded_dim_size)
        )
        self.pos_encoding.unsqueeze(0).transpose(0, 1).contiguous()

    def forward(self, src_input: Tensor) -> Tensor:
        return src_input + self.pos_encoding[:, 0]


class TransformerPositionWiseFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.n_heads = self.config.neural_net.n_heads
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.linear_one = nn.Linear(
            self.model_dimensions, self.config.neural_net.hidden_layers
        )
        self.linear_two = nn.Linear(
            self.config.neural_net.hidden_layers, self.model_dimensions
        )
        self.relu_layer = nn.ReLU()
        self.dropout_one = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.dropout_two = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(self, src_input) -> Tensor:
        output = self.linear_one(src_input)
        output = self.relu_layer(output)
        output = self.dropout_one(output)
        output = self.linear_two(output)
        return self.dropout_two(output)


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, embedded_dim_size: int) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.n_heads = self.config.neural_net.n_heads
        self.attention_output_size = embedded_dim_size // self.n_heads
        self.attention_layers = nn.ModuleList(
            [
                TransformerScaledDotProductAttention(
                    output_size=self.attention_output_size,
                    tokenizer=tokenizer,
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
    def __init__(
        self, output_size, tokenizer: AutoTokenizer, embedded_dim_size: int
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.output_size = output_size
        self.pad_id = tokenizer.pad_token_id
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
        batch_size = query.size()[0]
        tgt_len = query.size()[1]
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
        # Matmul of our values and final attention score.
        return bmm(input=attention_score, mat2=value)
