import time
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from src.agent_config import get_agent_config
from src.api_interface import DattaBotAPIResponse
from src.logger import get_logger
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer
from src.util import get_logging_level_from_config, is_device_cpu, is_rank_0
from torch import nn, Tensor
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel


class DattaBotInferenceEngine:
    """
    High-performance inference engine for DattaBot model.

    Features:
    - Multi-GPU support (tensor parallelism via DDP)
    - Efficient batched generation
    - Multiple sampling strategies (greedy, top-k, top-p, beam search)
    - KV caching for autoregressive generation
    - AdaptiveLogSoftmax handling
    """

    def __init__(
        self,
        model: Union[DattaBotModel, DistributedDataParallel],
        adaptive_softmax: nn.AdaptiveLogSoftmaxWithLoss,
        device: Optional[str] = None,
    ):
        """
        Initialize inference engine.
        Args:
            model: The DattaBot model or DDP-wrapped model
            device: Device to run inference on
        """
        self.agent_config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.agent_config)
        )
        # Validate inference config exists
        assert hasattr(
            self.agent_config, "inference"
        ), "Config must have 'inference' section with inference parameters"
        assert hasattr(
            self.agent_config.inference, "max_new_tokens"
        ), "Config must have inference.max_new_tokens"
        assert hasattr(
            self.agent_config.inference, "temperature"
        ), "Config must have inference.temperature"
        assert hasattr(
            self.agent_config.inference, "top_k"
        ), "Config must have inference.top_k"
        assert hasattr(
            self.agent_config.inference, "top_p"
        ), "Config must have inference.top_p"
        assert hasattr(
            self.agent_config.inference, "do_sample"
        ), "Config must have inference.do_sample"
        assert hasattr(
            self.agent_config.inference, "num_beams"
        ), "Config must have inference.num_beams"
        # Set device
        if device is None:
            self.device = self.agent_config.env.device
            if dist.is_available() and dist.is_initialized():
                self.device = f"cuda:{dist.get_rank()}"
        else:
            self.device = device
        self.model = model
        self.model.eval()
        # Setup tokenizer
        self.tokenizer = get_tokenizer(encoding_name="o200k_harmony")
        # Model parameters
        self.d_model = self.agent_config.neural_net.model_dimensions
        self.vocab_size = self.tokenizer.vocab_size
        self.max_response_tokens = self.agent_config.agent.max_response_tokens
        # Setup AdaptiveLogSoftmax for inference
        # Must match the training configuration
        assert hasattr(
            self.agent_config.neural_net, "model_dimensions"
        ), "Config must have neural_net.model_dimensions"
        self.adaptive_softmax = adaptive_softmax
        self.logger.info(
            f"Inference engine initialized on {self.device} with vocab_size={self.vocab_size}"
        )
        self.logger.info(
            f"Inference config: max_new_tokens={self.agent_config.inference.max_new_tokens}, "
            f"temperature={self.agent_config.inference.temperature}, "
            f"top_k={self.agent_config.inference.top_k}, "
            f"top_p={self.agent_config.inference.top_p}, "
            f"do_sample={self.agent_config.inference.do_sample}, "
            f"num_beams={self.agent_config.inference.num_beams}"
        )
        # KV cache for efficient autoregressive generation
        self.kv_cache: Optional[dict] = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        **kwargs,
    ) -> List[DattaBotAPIResponse]:
        """
        Generate sequences from input_ids.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search

        Returns:
            List of DattaBotAPIResponse objects (one per batch item)
        """
        start_time = time.time()
        # Get config values with overrides
        max_new_tokens = max_new_tokens or self.agent_config.inference.max_new_tokens
        temperature = (
            temperature
            if temperature is not None
            else self.agent_config.inference.temperature
        )
        top_k = top_k or self.agent_config.inference.top_k
        top_p = top_p if top_p is not None else self.agent_config.inference.top_p
        do_sample = (
            do_sample
            if do_sample is not None
            else self.agent_config.inference.do_sample
        )
        num_beams = num_beams or self.agent_config.inference.num_beams
        # Validate inputs
        assert input_ids.dim() == 2, f"Expected 2D input, got {input_ids.dim()}D"
        batch_size, input_seq_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        self.logger.debug(
            f"Generating {max_new_tokens} tokens for batch_size={batch_size}, "
            f"input_seq_len={input_seq_len}"
        )
        # Store original input for encodings
        original_input_ids = input_ids.clone()
        # Choose generation strategy
        if num_beams > 1:
            generated_ids = self._beam_search(
                input_ids, max_new_tokens, num_beams, temperature
            )
        else:
            generated_ids = self._greedy_or_sample(
                input_ids, max_new_tokens, temperature, top_k, top_p, do_sample
            )
        # Decode to text
        generated_text_list = self.tokenizer.decode(generated_ids.tolist())
        # Calculate metrics
        inference_time = time.time() - start_time
        total_tokens = generated_ids.numel()
        tokens_per_second = total_tokens / inference_time if inference_time > 0 else 0
        if is_rank_0():
            self.logger.info(
                f"Generated {total_tokens} tokens in {inference_time:.2f}s "
                f"({tokens_per_second:.2f} tokens/s)"
            )
        responses: list[DattaBotAPIResponse] = []
        for i in range(batch_size):
            # Calculate per-item tokens
            item_tokens = generated_ids[i].numel()
            item_new_tokens = generated_ids[i].shape[0] - original_input_ids[i].shape[0]

            response = DattaBotAPIResponse(
                response_dict={
                    "output_text": generated_text_list[i],
                    "choices": [
                        {
                            "text": generated_text_list[i],
                            "index": 0,
                            "finish_reason": (
                                "length"
                                if generated_ids[i].shape[0] >= self.max_response_tokens
                                else "stop"
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": original_input_ids[i].shape[0],
                        "completion_tokens": item_new_tokens,
                        "total_tokens": item_tokens,
                    },
                },
                metadata={
                    "tensor_response": generated_ids[i].clone().detach(),
                    "tokenizer_encodings": [original_input_ids[i].tolist()],
                    "tokenizer_decodings": [generated_text_list[i]],
                    "inference_time": inference_time / batch_size,
                    "tokens_per_second": tokens_per_second,
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "do_sample": do_sample,
                        "num_beams": num_beams,
                    },
                    "model_info": {
                        "d_model": self.d_model,
                        "vocab_size": self.vocab_size,
                        "device": str(self.device),
                    },
                },
            )
            responses.append(response)
        return responses

    def _greedy_or_sample(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> Tensor:
        """
        Greedy decoding or sampling-based generation.

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            do_sample: Whether to sample

        Returns:
            generated_ids: [batch, total_seq_len]
        """
        batch_size, cur_len = input_ids.shape
        generated = input_ids.clone()

        # Track which sequences are finished
        unfinished = torch.ones(batch_size, dtype=torch.long, device=self.device)

        for step in range(max_new_tokens):
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device,
                    enabled=(not is_device_cpu(self.device)),
                    dtype=torch.bfloat16,
                ):
                    # Forward pass through model
                    logits = self.model(generated)  # [batch, seq_len, d_model]
                # Get logits for last position
                last_logits = logits[:, -1, :]  # [batch, d_model]
                # Project to vocabulary using adaptive softmax
                next_token_logits = self._project_to_vocab(
                    last_logits
                )  # [batch, vocab_size]
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Get next token
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        next_token_logits = self._top_k_filtering(
                            next_token_logits, top_k
                        )

                    # Apply nucleus (top-p) filtering
                    if top_p < 1.0:
                        next_token_logits = self._top_p_filtering(
                            next_token_logits, top_p
                        )

                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Update generated sequence
                generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token (if tokenizer has one)
            if hasattr(self.tokenizer, "eos_token_id"):
                unfinished = unfinished.mul(
                    next_token.squeeze(-1).ne(self.tokenizer.eos_token_id).long()
                )

                # Stop if all sequences are finished
                if unfinished.max() == 0:
                    self.logger.debug(f"All sequences finished at step {step}")
                    break

            # Check max length
            if generated.shape[1] >= self.max_response_tokens:
                self.logger.debug(f"Reached max_response_tokens at step {step}")
                break

        return generated

    def _beam_search(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        num_beams: int,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Beam search generation.

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            num_beams: Number of beams
            temperature: Sampling temperature

        Returns:
            generated_ids: [batch, total_seq_len]
        """
        batch_size, cur_len = input_ids.shape

        # Expand input for beam search: [batch * num_beams, seq_len]
        input_ids_expanded = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        input_ids_expanded = input_ids_expanded.view(batch_size * num_beams, cur_len)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)  # [batch * num_beams]

        generated = input_ids_expanded.clone()

        for step in range(max_new_tokens):
            # Forward pass
            logits = self.model(generated)  # [batch * num_beams, seq_len, d_model]
            last_logits = logits[:, -1, :]  # [batch * num_beams, d_model]

            # Project to vocabulary
            next_token_logits = self._project_to_vocab(last_logits)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Get log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            next_token_scores = next_token_scores + beam_scores[:, None]

            # Reshape to [batch, num_beams * vocab_size]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * self.vocab_size
            )

            # Get top 2 * num_beams scores
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            # Process each batch
            next_batch_beam = []
            for batch_idx in range(batch_size):
                # Beam indices
                beam_idx = next_tokens[batch_idx] // self.vocab_size
                token_idx = next_tokens[batch_idx] % self.vocab_size

                # Select top num_beams
                for rank, (score, beam, token) in enumerate(
                    zip(next_token_scores[batch_idx], beam_idx, token_idx)
                ):
                    if rank >= num_beams:
                        break
                    next_batch_beam.append((score, batch_idx * num_beams + beam, token))

            # Update beam scores and sequences
            beam_scores = torch.tensor(
                [x[0] for x in next_batch_beam], device=self.device
            )
            beam_idx = torch.tensor(
                [x[1] for x in next_batch_beam], dtype=torch.long, device=self.device
            )
            next_tokens = torch.tensor(
                [x[2] for x in next_batch_beam], dtype=torch.long, device=self.device
            )

            # Update generated sequences
            generated = generated[beam_idx]
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

        # Return best beam for each batch
        generated = generated.view(batch_size, num_beams, -1)
        return generated[:, 0, :]  # [batch, seq_len]

    def _project_to_vocab(self, hidden_states: Tensor) -> Tensor:
        """
        Project hidden states to vocabulary logits using adaptive softmax.

        Args:
            hidden_states: [batch, d_model]

        Returns:
            logits: [batch, vocab_size]
        """
        # Use the adaptive softmax log_prob method
        # We need to get full vocabulary distribution
        batch_size = hidden_states.size(0)
        # Get logits for all clusters
        head_logits = hidden_states @ self.adaptive_softmax.head.weight.t()
        if self.adaptive_softmax.head.bias is not None:
            head_logits = head_logits + self.adaptive_softmax.head.bias
        # Initialize full logits tensor
        logits = torch.zeros(
            batch_size, self.vocab_size, device=self.device, dtype=hidden_states.dtype
        )
        # Fill in head cluster (most frequent words)
        cutoff = self.adaptive_softmax.cutoffs[0]
        logits[:, :cutoff] = head_logits[:, :cutoff]
        # Process tail clusters
        for i, (start_idx, end_idx) in enumerate(
            zip(self.adaptive_softmax.cutoffs[:-1], self.adaptive_softmax.cutoffs[1:])
        ):
            cluster_idx = cutoff + i
            cluster_logit = head_logits[:, cluster_idx : cluster_idx + 1]
            # Get tail projection
            # tail is a Sequential module with two Linear layers
            tail = self.adaptive_softmax.tail[i]
            # Pass through the Sequential module
            tail_logits = tail(hidden_states)
            # Combine cluster probability with tail probabilities
            logits[:, start_idx:end_idx] = cluster_logit + tail_logits
        return logits

    def _top_k_filtering(
        self, logits: Tensor, top_k: int, filter_value: float = -float("Inf")
    ) -> Tensor:
        """
        Filter a distribution using top-k filtering.

        Args:
            logits: [batch, vocab_size]
            top_k: Keep only top k tokens
            filter_value: Value to set for filtered tokens

        Returns:
            filtered_logits: [batch, vocab_size]
        """
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    def _top_p_filtering(
        self, logits: Tensor, top_p: float, filter_value: float = -float("Inf")
    ) -> Tensor:
        """
        Filter a distribution using nucleus (top-p) filtering.

        Args:
            logits: [batch, vocab_size]
            top_p: Cumulative probability threshold
            filter_value: Value to set for filtered tokens

        Returns:
            filtered_logits: [batch, vocab_size]
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        return logits

    @torch.no_grad()
    def batch_generate(
        self, queries: list[str], voting_strategy: str = "majority", **generation_kwargs
    ) -> list[DattaBotAPIResponse]:
        """
        Generate responses on all GPUs (with different random seeds) and ensemble.

        Args:
            queries: List of input texts
            voting_strategy: 'majority', 'first', or 'longest'
            **generation_kwargs: Arguments to pass to generate()
        Returns:
            List of DattaBotAPIResponse objects (ensembled results on rank 0)
        """
        if not queries:
            return []

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Set different random seed per GPU (for diverse sampling)
        if generation_kwargs.get("do_sample", True):
            torch.manual_seed(42 + rank)

        # All GPUs generate responses
        input_ids_list = [self.tokenizer.encode(text) for text in queries]
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_response_tokens)

        padded_ids = []
        for ids in input_ids_list:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_ids.append(ids)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        local_responses = self.generate(input_ids, **generation_kwargs)

        if not dist.is_initialized():
            return local_responses

        # Gather all responses to rank 0
        all_responses_per_rank = [None] * world_size
        dist.gather_object(
            local_responses, all_responses_per_rank if rank == 0 else None, dst=0
        )

        if rank == 0:
            # Ensemble responses
            final_responses = []
            for query_idx in range(len(queries)):
                # Get all GPU responses for this query
                candidates = [
                    responses[query_idx].text for responses in all_responses_per_rank
                ]

                if voting_strategy == "majority":
                    # Vote on most common response
                    from collections import Counter

                    most_common = Counter(candidates).most_common(1)[0][0]
                    final_text = most_common
                elif voting_strategy == "longest":
                    # Use longest response
                    final_text = max(candidates, key=len)
                else:  # "first" or default
                    # Use rank 0 response
                    final_text = candidates[0]

                # Use rank 0's metadata but with ensembled text
                final_responses.append(
                    DattaBotAPIResponse(
                        response_dict={"output_text": final_text},
                        metadata=all_responses_per_rank[0][query_idx].metadata,
                    )
                )

            return final_responses
        else:
            return []
