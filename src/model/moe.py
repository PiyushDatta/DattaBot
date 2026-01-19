from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agent_config import get_agent_config
from src.model.rms_norm import RMSNorm


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
            torch.empty(self.num_experts, self.hidden_dim, self.d_model)
        )
        # First fully connected (input) layer biases for all experts: [num_experts, hidden_dim]
        self.expert_fc1_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim)
        )
        # Second layer weights: W2 ∈ R^(num_experts × d_model × hidden_dim)
        self.expert_fc2_weight = nn.Parameter(
            torch.empty(self.num_experts, self.d_model, self.hidden_dim)
        )
        # Second fully connected (output) layer biases for all experts: [num_experts, d_model]
        self.expert_fc2_bias = nn.Parameter(torch.empty(self.num_experts, self.d_model))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        # Select top-k experts per token
        # expert/gate_scores: [num_tokens, num_experts_per_token] - unnormalized scores
        # expert_indices: [num_tokens, num_experts_per_token] - which experts to use
        gate_scores, expert_indices = torch.topk(
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
        # Memory-efficient loop-based approach: process one expert at a time
        # This avoids creating [num_tokens, num_experts_per_token, hidden, d_model] tensors
        num_tokens = tokens.shape[0]
        output = torch.zeros(
            num_tokens, self.d_model, device=tokens.device, dtype=tokens.dtype
        )

        # Loop over each expert
        for expert_idx in range(self.num_experts):
            # Find which token-slot pairs are routed to this expert
            # expert_mask: [num_tokens, num_experts_per_token] boolean mask
            expert_mask = expert_indices == expert_idx
            # Skip if no tokens are routed to this expert
            if not expert_mask.any():
                continue
            # Get the routing weights for tokens routed to this expert
            # masked_weights: [num_tokens, num_experts_per_token] with 0s where not routed
            masked_weights = routing_weights * expert_mask.float()
            # Sum across expert slots to get per-token weight for this expert
            # token_weights: [num_tokens]
            token_weights = masked_weights.sum(dim=1)
            # Find tokens that have non-zero weight for this expert
            # active_mask: [num_tokens] boolean
            active_mask = token_weights > 0
            if not active_mask.any():
                continue
            # Get the tokens that are routed to this expert
            # active_tokens: [num_active, d_model]
            active_tokens = tokens[active_mask]
            # active_weights: [num_active, 1]
            active_weights = token_weights[active_mask].unsqueeze(1)

            # Apply expert MLP: E_i(x) = W2_i * activation(W1_i * x + b1_i) + b2_i
            # fc1: [num_active, hidden_dim]
            hidden = F.linear(
                active_tokens,
                self.expert_fc1_weight[expert_idx],
                self.expert_fc1_bias[expert_idx],
            )
            hidden = self.activation(hidden)
            hidden = self.dropout1(hidden)

            # fc2: [num_active, d_model]
            expert_out = F.linear(
                hidden,
                self.expert_fc2_weight[expert_idx],
                self.expert_fc2_bias[expert_idx],
            )
            expert_out = self.dropout2(expert_out)

            # Accumulate weighted output: G(x)_i * E_i(x)
            output[active_mask] += active_weights * expert_out
        # Reshape back to [batch_size, seq_len, d_model]
        output = output.view(batch_size, seq_len, self.d_model)
        return x + output

    def _compute_load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
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

    def get_load_balancing_loss(self) -> Optional[torch.Tensor]:
        """
        Returns the auxiliary load balancing loss.
        This should be added to the main loss during training:
            total_loss = main_loss + alpha * aux_loss

        Where alpha is typically 0.01
        """
        return self.aux_loss
