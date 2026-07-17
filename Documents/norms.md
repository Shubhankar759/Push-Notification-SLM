**File-level: what goes in, what comes out**
Hidden states of shape `(batch_size, seq_len, hidden_size)` go in, RMS-normalized hidden states of the identical shape come out. No shape change, one learned parameter, no bias.

---

**`RMSNorm` class variables**
- `eps` — small constant added under the square root for numerical stability (`1e-6` from config)
- `weight` — `nn.Parameter` of shape `(dim,)`, initialized to all ones (i.e. starts as a no-op scale)

---

**`forward(x)`**
- Takes: `(batch_size, seq_len, hidden_size)`
- Computes `rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)` — per-token root-mean-square over the hidden dimension
- Returns `weight * (x / rms)` — same shape as input

---

**Flagged for your review**
1. Your locked convention (from earlier sessions) specifies RMSNorm should do an **explicit float32 upcast internally** before the mean/sqrt, then cast back down. This implementation does the math in whatever dtype `x` arrives in. It's harmless right now since you're running everything in fp32, but the moment you introduce mixed precision (fp16/bf16 autocast) during training, computing `mean(x.pow(2))` in half precision is a classic source of RMSNorm instability (overflow/underflow in the squared term). This is the one thing I'd fix before turning on `torch.cuda.amp` — happy to patch it in when you say go.
2. This version doesn't distinguish "Pre-Norm placement" architecturally — that's actually handled correctly at the call site in `TransformerBlock` (norm happens before attention/FFN, not after), so no issue there, just noting it's an external-to-this-file guarantee, not something `RMSNorm` itself enforces.
