import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    """
    Back-half of the attention sub-layer: takes already-projected,
    already-RoPE-rotated Q/K (V unrotated) and produces the block's
    attention output, ready for the residual add.

    Does NOT do: QKV linear projections (QKVProjection), RoPE (RotaryEmbedding),
    or normalization (RMSNorm, applied before this module at the block level).
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_groups,
        head_dim,
        max_seq_len,
        init_std=0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.q_per_kv = num_heads // num_kv_groups

        self.W_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Precomputed causal mask, cached the same way RoPE caches cos/sin:
        # a non-persistent buffer that travels with the model to GPU but is
        # never trained and never saved in the state_dict.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self._init_weights(init_std)

    def _init_weights(self, std):
        nn.init.normal_(self.W_o.weight, mean=0.0, std=std)

    def _expand_kv(self, x):
        # x: (batch, num_kv_groups, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim), each kv head repeated
        # q_per_kv times contiguously (kv head 0 -> q heads 0..q_per_kv-1, etc.)
        batch_size, num_kv_groups, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(batch_size, num_kv_groups, self.q_per_kv, seq_len, head_dim)
        x = x.reshape(batch_size, num_kv_groups * self.q_per_kv, seq_len, head_dim)
        return x

    def forward(self, q, k, v):
        # q: (batch, num_heads, seq_len, head_dim) — RoPE-rotated
        # k, v: (batch, num_kv_groups, seq_len, head_dim) — k RoPE-rotated, v not
        batch_size, num_heads, seq_len, head_dim = q.shape

        k = self._expand_kv(k)  # (batch, num_heads, seq_len, head_dim)
        v = self._expand_kv(v)  # (batch, num_heads, seq_len, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, seq_len)

        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, num_heads * head_dim)
        )

        output = self.W_o(attn_output)  # (batch, seq_len, hidden_size)
        return output