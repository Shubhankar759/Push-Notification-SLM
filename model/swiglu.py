"""
swiglu_ffn.py

SwiGLU Feed-Forward sub-layer for the decoder block.

Sits after Pre-Norm RMSNorm, in parallel role to the attention sub-layer:
    RMSNorm -> Attention -> residual add
    RMSNorm -> SwiGLUFeedForward -> residual add   <-- this module

Three weight matrices, no bias (bias=False, consistent with QKV projections):
    W_gate : hidden_size -> intermediate_size   (gated branch, Swish/SiLU applied)
    W_up   : hidden_size -> intermediate_size   (value branch, no activation)
    W_down : intermediate_size -> hidden_size   (projects back down)

forward:
    gate = SiLU(W_gate(x))
    up   = W_up(x)
    out  = W_down(gate * up)

No sequence-position or inter-token logic here — purely per-token, applied
identically and independently at every position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU-gated feed-forward block.

    Args:
        hidden_size (int): model working dimension. Placeholder default = 512,
            matching the rest of the project's locked placeholder constants.
            Pass a different value at construction time to override — nothing
            here is hardcoded beyond the default argument, so changing the
            project-wide hidden_size later only means changing what's passed in.
        intermediate_size (int, optional): width of the gated hidden space.
            If not provided, it is derived from hidden_size as
            round(8/3 * hidden_size), per Llama/PaLM SwiGLU convention, with
            no further rounding to a hardware-friendly multiple (exact 8/3
            value, per project decision).
        init_std (float): standard deviation for normal weight init (0.02,
            same convention as token embeddings and QKV projections).
    """

    def __init__(self, hidden_size: int = 512, intermediate_size: int = None, init_std: float = 0.02):
        super().__init__()

        self.hidden_size = hidden_size
        # Derived, not independently chosen, unless explicitly overridden.
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None
            else round(hidden_size * 8 / 3)
        )
        self.init_std = init_std

        self.W_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self._init_weights(self.init_std)

    def _init_weights(self, std: float):
        """Normal init (mean 0, std=0.02) on all three matrices. No biases exist to zero."""
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=std)
        nn.init.normal_(self.W_up.weight, mean=0.0, std=std)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size), already RMSNorm-normalized.

        Returns:
            (batch_size, seq_len, hidden_size) — residual add happens outside
            this module, at the decoder-block level.
        """
        gate = F.silu(self.W_gate(x))   # (batch, seq_len, intermediate_size)
        up = self.W_up(x)               # (batch, seq_len, intermediate_size)
        gated = gate * up               # element-wise gating, (batch, seq_len, intermediate_size)
        out = self.W_down(gated)        # (batch, seq_len, hidden_size)
        return out