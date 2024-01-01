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

        self.encoder_stack = TransformerEncoderStack(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            model_dimensions=self.model_dimensions,
        )

    def forward(self, src_input) -> Tensor:
        encoder_stack_output = self.encoder_stack(src_input)
        return encoder_stack_output


class TransformerEncoderStack(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, model_dimensions: int) -> None:
        super().__init__()
        self.logger = get_logger()
        self.config = get_agent_config()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.model_dimensions = model_dimensions
        self.multi_head_attn = TransformerMultiHeadAttention()
        self.position_wise_ffn = TransformerPositionWiseFeedForward()
        self.norm_one = nn.LayerNorm(self.model_dimensions)

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
        query = self.split(query, self.n_heads)
        key = self.split(key, self.n_heads)
        value = self.split(value, self.n_heads)
        # Call scaled dot-product attention layer.
        attention_score = self.attention_layer(
            query=query, key=key, value=value, mask=None
        )
        # Concat.
        output = self.concat(attention_score)
        # Linear and return output.
        return self.concat_layer(output)

    def split(self, input_tensor: Tensor):
        self.logger.info(f"Splitting this tensor: {input_tensor}")
        self.logger.info(input_tensor.size())
        batch_size, n_heads, length, model_dimensions = input_tensor.size()
        model_depth = model_dimensions // n_heads
        return tensor.view(batch_size, length, n_heads, model_depth).transpose(1, 2)

    def concat(self, input_tensor: Tensor):
        """
        Opposite of split.
        """
        batch_size, n_heads, length, model_depth = input_tensor.size()
        model_dimensions = model_depth * n_heads
        return tensor.transpose(1, 2).contiguous.view(
            batch_size, length, n_heads, model_dimensions
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
        # Tensor input is 4D tensor.
        _batch_size, _n_heads, _length, model_dimensions = key.size()
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
