**File-level: what goes in, what comes out**
Already-projected, already-RoPE-rotated Q `(batch_size, num_heads, seq_len, head_dim)` and RoPE-rotated K / un-rotated V `(batch_size, num_kv_groups, seq_len, head_dim)` go in. A single tensor `(batch_size, seq_len, hidden_size)` comes out — the finished attention sub-layer output, ready to be added back into the residual stream by the enclosing decoder block. This module does not do QKV projection (QKVProjection), RoPE (RotaryEmbedding), or normalization (RMSNorm) — it only does the attention computation itself.

---

**`GQAAttention` class variables**
- `hidden_size` — model working dimension (512 placeholder)
- `num_heads` — number of Q attention heads (8 placeholder)
- `num_kv_groups` — number of K/V heads (2 placeholder)
- `head_dim` — size of each head vector, `hidden_size // num_heads = 64`
- `max_seq_len` — longest sequence the cached causal mask supports (128 placeholder); passed in at construction, same convention as RoPE's cached cos/sin tables
- `q_per_kv` — how many Q heads share one K/V head, `num_heads // num_kv_groups = 4`
- `W_o` — `nn.Linear(num_heads * head_dim, hidden_size, bias=False)`, the output projection
- `causal_mask` — `(max_seq_len, max_seq_len)` boolean buffer, upper triangle (excluding diagonal) set `True` where attention must be blocked; registered as a non-persistent buffer exactly like RoPE's cos/sin — travels to GPU with the model, never trained, never checkpointed, sliced down to the current `seq_len` at forward time

---

**`GQAAttention` methods**
- `_init_weights(std)` — normal init (mean 0, std 0.02) on `W_o`; nothing in, nothing out
- `_expand_kv(x)` — takes a K or V tensor `(batch, num_kv_groups, seq_len, head_dim)`, repeats each kv head `q_per_kv` times contiguously, returns `(batch, num_heads, seq_len, head_dim)`. Pure reshape/repeat, no new parameters — this is where GQA's cache savings come from (only the small, unexpanded K/V need to be stored; expansion happens on the fly)
- `forward(q, k, v)` — runs the full attention computation described below, returns `(batch_size, seq_len, hidden_size)`

---

**Inside `forward()` — step by step**
1. Read `batch_size, num_heads, seq_len, head_dim` off Q's shape
2. Expand K and V from `num_kv_groups` heads to `num_heads` heads via `_expand_kv`
3. Compute raw attention scores: `Q @ K.transpose(-2, -1)`, scaled by `1/sqrt(head_dim)` → `(batch, num_heads, seq_len, seq_len)`
4. Slice the cached `causal_mask` down to `[:seq_len, :seq_len]` and use it to set future positions to `-inf` via `masked_fill`
5. Softmax across the key dimension (last dim) to get attention weights
6. Weighted sum: `attn_weights @ V` (expanded) → `(batch, num_heads, seq_len, head_dim)`
7. Transpose heads and seq_len back, then reshape/concatenate heads into `(batch, seq_len, num_heads * head_dim)`, which equals `(batch, seq_len, hidden_size)` since `head_dim = hidden_size // num_heads`
8. Apply `W_o` to mix information across heads
9. Return the result

---

**Parameter cost**
Only one new weight matrix versus what QKVProjection already accounts for: `W_o`, contributing `hidden_size * hidden_size` parameters. The K/V head-expansion step adds zero parameters — it's pure reshaping/repetition, which is exactly what makes GQA's memory savings possible in deployment (only the smaller, unexpanded K/V tensors need caching at inference time).

---

**Initialization note**
`W_o` uses the same convention as every other projection in this project: normal distribution, mean 0, std 0.02 (`init_std`), `bias=False`.

---

**Design choice flagged during this build**
The causal mask is precomputed once at construction time (size `max_seq_len x max_seq_len`) and cached as a non-persistent buffer, then sliced per forward call — mirroring how `RotaryEmbedding` caches its `cos`/`sin` tables, rather than rebuilding the mask fresh on every forward pass. This means `GQAAttention.__init__` now requires `max_seq_len` as an input, in addition to `hidden_size`, `num_heads`, `num_kv_groups`, and `head_dim`.

---

**Open items carried forward (unchanged from handoff)**
- Exact `hidden_size`, `num_layers`, `num_attention_heads`, `num_kv_groups` still not finalized
- Stage 1 pretraining corpus source and size still undecided

**Next logical component:** custom `RMSNorm(nn.Module)`, then full decoder block assembly wiring RMSNorm → QKV Projection → RoPE → GQAAttention → residual add → RMSNorm → SwiGLU → residual add, then stacking `num_layers` of these blocks.
