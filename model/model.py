"""
Full SLM Model — wires your ready modules together
====================================================

Pipeline per your architecture:

    token_ids
      -> Embedding
      -> [ RMSNorm -> QKVProjection -> RoPE -> GQAAttention -> residual add
          -> RMSNorm -> SwiGLU FFN            -> residual add ]  x num_layers
      -> final RMSNorm
      -> lm_head (optionally tied to embedding weight)
      -> logits

*** ASSUMPTIONS — ADJUST THESE IMPORT LINES / CALL SIGNATURES TO MATCH
    YOUR ACTUAL FILES. I only have your qkv_projections.py; the other
    three are assumed to have the interfaces below. If your real
    modules differ, paste them and I'll rewire this exactly. ***

Assumed interfaces:
    RMSNorm(dim, eps=1e-6).forward(x) -> x_normed                (same shape)
    QKVProjection(hidden_size, num_heads, num_kv_groups, bias).forward(x)
        -> (Q, K, V)   each (batch, heads, seq_len, head_dim)
    GQAAttention(hidden_size, num_heads, num_kv_groups).forward(Q, K, V, causal=True)
        -> attn_out  (batch, seq_len, hidden_size)   <- already merged
                                                          heads + output-projected
        (assumed to apply RoPE internally — flag this if it doesn't;
         if RoPE is a separate module you already have, tell me and
         I'll insert it explicitly between QKVProjection and attention)
    SwiGLUFFN(hidden_size, ffn_dim, bias=False).forward(x) -> x    (same shape)
"""

import torch
import torch.nn as nn

from qvk_projections import QKVProjection

# ---- ADJUST THESE THREE IMPORTS TO MATCH YOUR ACTUAL FILES ----
from norms import RMSNorm
from attention import GQAAttention
from swiglu import SwiGLUFeedForward
# -----------------------------------------------------------------

from config import SLMConfig


def compute_ffn_dim(hidden_size: int, multiplier: float) -> int:
    """
    Standard SwiGLU sizing: raw multiplier is applied, then rounded to
    nearest multiple of 8 for hardware-friendly matmul shapes (common
    convention from LLaMA / Mistral configs).
    """
    raw = int(hidden_size * multiplier)
    return ((raw + 7) // 8) * 8


class TransformerBlock(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.qkv = QKVProjection(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_groups=cfg.num_kv_groups,
            bias=cfg.bias,
            init_std=cfg.init_std,
        )
        self.attn = GQAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_groups=cfg.num_kv_groups,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            init_std=cfg.init_std
            
        )

        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        ffn_dim = compute_ffn_dim(cfg.hidden_size, cfg.ffn_multiplier)
        self.ffn = SwiGLUFeedForward(
        hidden_size=cfg.hidden_size,
        intermediate_size=ffn_dim,  # or just leave None to use its own 8/3 default
        init_std=cfg.init_std,
        )

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Attention sub-layer (pre-norm + residual) ---
        residual = x
        h = self.attn_norm(x)
        Q, K, V = self.qkv(h)
        attn_out = self.attn(Q, K, V)
        x = residual + self.dropout(attn_out)

        # --- FFN sub-layer (pre-norm + residual) ---
        residual = x
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        x = residual + self.dropout(ffn_out)

        return x


class SLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)

        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.init_std)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) long tensor
        returns logits: (batch, seq_len, vocab_size)
        """
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def num_params(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(
                p.numel() for n, p in self.named_parameters()
                if "embed" not in n and "lm_head" not in n
            )
        return sum(p.numel() for p in self.parameters())
