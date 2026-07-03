**File-level: what goes in, what comes out** - RMSNorm-normalized hidden states `(batch_size, seq_len, hidden_size)` go in, transformed hidden states of the same shape `(batch_size, seq_len, hidden_size)` come out. Residual add happens outside this module, at the decoder-block level.

---

**`SwiGLUFeedForward` class variables**
- `hidden_size` — model working dimension (512 placeholder, overridable at construction — nothing hardcoded beyond the default arg, so future changes just mean passing a new value in)
- `intermediate_size` — width of the gated space; if not passed explicitly, derived as `round(hidden_size * 8/3)`, no further rounding to a hardware multiple (per your call — exact 8/3, only rounded to the nearest int since dims must be integers)
- `init_std` — 0.02, same convention as everywhere else
- `W_gate` — `nn.Linear(hidden_size, intermediate_size, bias=False)`, the branch that gets Swish/SiLU
- `W_up` — `nn.Linear(hidden_size, intermediate_size, bias=False)`, the branch that gets multiplied against the gated one, no activation
- `W_down` — `nn.Linear(intermediate_size, hidden_size, bias=False)`, projects back down

---

**Methods**
- `_init_weights(std)` — normal init, mean 0, std 0.02, on all three matrices; no biases exist since `bias=False` everywhere
- `forward(x)` — takes `(batch_size, seq_len, hidden_size)`, returns `(batch_size, seq_len, hidden_size)`

**Inside `forward()`:**
- `gate = SiLU(W_gate(x))` — shape `(batch, seq_len, intermediate_size)`
- `up = W_up(x)` — shape `(batch, seq_len, intermediate_size)`
- `gated = gate * up` — element-wise multiply, the actual gating step
- `out = W_down(gated)` — back to `(batch, seq_len, hidden_size)`

One thing worth flagging: since `intermediate_size` is currently derived at `hidden_size=512` → `round(512 * 8/3) = 1365`, not a multiple of 64/128. That's fine numerically (Linear layers don't care), but if you ever quantize to int4/int8 for GGUF export, odd intermediate dims can occasionally hit alignment inefficiencies on some mobile kernels. Not blocking anything now — just flagging it since export-compatibility is a first-class constraint for you. Let me know if you want to revisit rounding once `hidden_size` is actually locked.