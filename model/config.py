"""
SLM Configuration
=================
Tune hidden_size / num_layers here and re-run driver.py's param count
until you land near your 15M target. Starting point below is a rough
guess for vocab_size=1000 — expect to adjust.
"""

from dataclasses import dataclass


@dataclass
class SLMConfig:
    vocab_size: int = 1000
    hidden_size: int = 384
    num_layers: int = 10
    num_heads: int = 8
    num_kv_groups: int = 2
    ffn_multiplier: float = 8 / 3   # standard SwiGLU expansion ratio (rounded internally)
    max_seq_len: int = 512
    rms_eps: float = 1e-6
    bias: bool = False
    init_std: float = 0.02
    tie_embeddings: bool = True     # share embedding & lm_head weights (saves params)
    dropout: float = 0.0
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
