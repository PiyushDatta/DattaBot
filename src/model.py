from torch import (
    nn,
    tensor,
    matmul,
    Tensor,
    set_default_device as torch_set_default_device,
    set_default_dtype as torch_set_default_dtype,
    float64 as torch_float64,
)
from math import sqrt as math_sqrt
from sentencepiece import SentencePieceProcessor
from src.logger import get_logger
from src.agent_config import get_agent_config


# Transformer Model.
class DattaBotModel(nn.Module):
    def __init__(self, tokenizer: SentencePieceProcessor) -> None:
        super().__init__()
        self.logger = get_logger()
        self.config = get_agent_config()
        self.device = self.config.env.device
        torch_set_default_device(self.device)
        torch_set_default_dtype(torch_float64)
        # Assumes tokenizer is already setup.
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.n_layers = self.config.neural_net.n_layers
        self.n_heads = self.config.neural_net.n_heads
        self.model_dimensions = self.config.neural_net.model_dimensions
        # Max tokens for response.
        self.response_max_tokens = self.config.agent.max_tokens
        self.logger.debug(f"Max tokens: {self.response_max_tokens}")
        self.encoder_stack = TransformerEncoderStack(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            model_dimensions=self.model_dimensions,
            tokenizer=tokenizer,
        )

    def forward(self, src_input: Tensor) -> Tensor:
        return src_input
        # encoder_stack_output = self.encoder_stack(src_input)
        # return encoder_stack_output


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        model_dimensions: int,
        tokenizer: SentencePieceProcessor,
    ) -> None:
        super().__init__()
        self.logger = get_logger()
        self.config = get_agent_config()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.model_dimensions = model_dimensions

        # Embedding layer
        self.embedding_layer = TransformerEmbedding(tokenizer=tokenizer)
        self.multi_head_attn = TransformerMultiHeadAttention()
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.norm_one = nn.LayerNorm(self.model_dimensions)
        self.dropout_one = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)
        self.norm_two = nn.LayerNorm(self.model_dimensions)
        self.dropout_two = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(self, src_input: Tensor) -> Tensor:
        orig_input = src_input
        # Multi-head attention.
        output = self.multi_head_attn(
            input_query=src_input, input_key=src_input, input_value=src_input
        )
        # Add and norm (one).
        output = self.dropout_one(output)
        output = self.norm_one(output + orig_input)
        # Position-wise feed forward network.
        orig_input = output
        output = self.position_wise_ffn(output)
        # Add and norm (two).
        output = self.dropout_two(output)
        output = self.norm_two(output + orig_input)
        return output


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        tokenizer: SentencePieceProcessor,
    ) -> None:
        super().__init__()
        self.config = get_agent_config()
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # self.embeddings_table = nn.Embedding(self.vocab_size, self.model_dimensions)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, src_input: Tensor) -> Tensor:
        assert (
            src_input.ndim == 2
        ), f"Expected: (max token sequence length, batch size), got {src_input.shape}"
        orig_input = src_input
        return src_input


class TransformerMultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger()
        self.config = get_agent_config()
        self.n_heads = self.config.neural_net.n_heads
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.attention_layer = TransformerScaledDotProductAttention()
        self.query_layer = nn.Linear(self.model_dimensions, self.model_dimensions)
        self.key_layer = nn.Linear(self.model_dimensions, self.model_dimensions)
        self.value_layer = nn.Linear(self.model_dimensions, self.model_dimensions)
        self.concat_layer = nn.Linear(self.model_dimensions, self.model_dimensions)

    def forward(
        self,
        input_query: Tensor,
        input_key: Tensor,
        input_value: Tensor,
    ) -> Tensor:
        # Linear.
        query = self.query_layer(input_query)
        key = self.key_layer(input_key)
        value = self.value_layer(input_value)
        # Split tensors by number of heads.
        query = self.split(query)
        key = self.split(key)
        value = self.split(value)
        # Call scaled dot-product attention layer.
        attention_score = self.attention_layer(
            query=query, key=key, value=value, mask=None
        )
        # Concat.
        output = self.concat(attention_score)
        # Linear and return output.
        return self.concat_layer(output)

    def split(self, input_tensor: Tensor):
        self.logger.debug(f"Splitting this tensor: {input_tensor}")
        self.logger.debug(input_tensor.size())
        model_dimensions, batch_size = input_tensor.size()
        model_depth = model_dimensions // self.n_heads
        return input_tensor.view(model_depth, model_depth).transpose(1, 2)

    def concat(self, input_tensor: Tensor):
        """
        Opposite of split.
        """
        batch_size, model_depth = input_tensor.size()
        model_dimensions = model_depth * self.n_heads
        return input_tensor.transpose(1, 2).contiguous.view(
            batch_size, model_dimensions
        )


class TransformerScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = get_agent_config()
        # Softmax being applied on the last
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: int = None,
    ) -> Tensor:
        # Tensor input is 2D tensor.
        _batch_size, model_dimensions = key.size()
        # Transpose keys. K^T.
        key = key.transpose(2, 3)
        # Get dot product of queries with all keys.
        # Then divide by sqrt of model_dimensions (d_k).
        # This covers the first matmul and scale blocks inside ScaledDotProductAttention.
        dot_product_score = matmul(input=query, other=key) / math_sqrt(model_dimensions)
        # Apply masking (optional).
        if mask is not None:
            dot_product_score = dot_product_score.masked_fill(mask == 0, -10000)
        # Apply softmax.
        attention_score = self.softmax(dot_product_score)
        # Matmul of our values and final attention score.
        return matmul(input=value, other=attention_score)


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
        self.dropout = nn.Dropout(p=self.config.neural_net.zeroed_drop_probability)

    def forward(self, src_input) -> Tensor:
        output = self.linear_one(src_input)
        output = self.relu_layer(output)
        output = self.dropout(output)
        return self.linear_two(output)
