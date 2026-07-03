"""
Q/K/V Projection Module
========================

Component position in the model:

    TokenEmbedding -> RMSNorm -> [QKVProjection -> reshape into heads
                                  -> RoPE on Q and K -> GQA attention]

This module takes the RMSNorm-normalized embedding output and projects it
into three separate roles:

    Q (Query)  — what each token is looking for
    K (Key)    — what each token offers to others searching
    V (Value)  — what each token actually hands back if attended to

In standard Multi-Head Attention (MHA), Q, K, and V all have the same
number of heads. In Grouped-Query Attention (GQA) — your locked
architecture — Q keeps the full num_heads count, while K and V share a
smaller set of num_kv_groups heads. Multiple Q heads map onto each
K/V group, reducing the K/V parameter cost and KV-cache size at
inference time (important for on-device mobile deployment).

Defaults (placeholders — update when architecture dims are locked):
    hidden_size  = 512
    num_heads    = 8     (Q heads)
    num_kv_groups = 2    (K and V heads — each group is shared by
                          num_heads // num_kv_groups = 4 Q heads)
    head_dim     = hidden_size // num_heads = 64
                   NOTE: derived from num_heads, NOT num_kv_groups,
                   per your locked architecture decision.
    bias         = False (standard for modern transformers)
"""

import torch
import torch.nn as nn


class QKVProjection(nn.Module):
    """
    Projects normalized hidden states into Q, K, V tensors.

    Input:
        x: FloatTensor, shape (batch_size, seq_len, hidden_size)
           This is the output of RMSNorm, NOT raw token embeddings.

    Output:
        Q: FloatTensor, shape (batch_size, num_heads,    seq_len, head_dim)
        K: FloatTensor, shape (batch_size, num_kv_groups, seq_len, head_dim)
        V: FloatTensor, shape (batch_size, num_kv_groups, seq_len, head_dim)

    Q has num_heads heads (full).
    K and V have num_kv_groups heads (smaller — this is the GQA design).
    head_dim = hidden_size // num_heads for all three.

    After this module:
        - Q and K go into RoPE for positional rotation, then into
          attention score computation (Q @ K^T).
        - V never gets RoPE — it carries content, not position.
        - K and V are expanded (repeated) from num_kv_groups up to
          num_heads before the attention dot product, which happens
          inside the attention block (not here).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_groups ({num_kv_groups}), so that each KV "
                f"group maps to an equal number of Q heads."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = hidden_size // num_heads   # derived from num_heads,
                                                   # NOT num_kv_groups

        # How many Q heads share each single K/V head
        self.q_per_kv = num_heads // num_kv_groups

        # Q projects to the full attention dimension
        # K and V project to the smaller KV dimension
        q_dim  = num_heads    * self.head_dim   # = hidden_size
        kv_dim = num_kv_groups * self.head_dim  # < hidden_size in GQA

        # Three separate learned linear projections — no shared weights
        # between Q, K, and V. No activation function; these are pure
        # linear transformations of the hidden state.
        self.W_q = nn.Linear(hidden_size, q_dim,  bias=bias)
        self.W_k = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.W_v = nn.Linear(hidden_size, kv_dim, bias=bias)

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        # Same 0.02 std normal init as TokenEmbedding, matching
        # GPT-2 / Llama-style initialization conventions.
        for linear in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_len, hidden_size) — RMSNorm output

        Returns Q, K, V reshaped into per-head layout:
            Q: (batch_size, num_heads,     seq_len, head_dim)
            K: (batch_size, num_kv_groups, seq_len, head_dim)
            V: (batch_size, num_kv_groups, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections — still flat (no head split yet)
        # q_flat: (batch_size, seq_len, num_heads * head_dim)
        # k_flat: (batch_size, seq_len, num_kv_groups * head_dim)
        # v_flat: (batch_size, seq_len, num_kv_groups * head_dim)
        q_flat = self.W_q(x)
        k_flat = self.W_k(x)
        v_flat = self.W_v(x)

        # Reshape into per-head layout and move heads before seq_len.
        # This layout — (batch, heads, seq_len, head_dim) — is what
        # RoPE and the attention score computation expect.
        Q = q_flat.view(batch_size, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        K = k_flat.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        V = v_flat.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        return Q, K, V