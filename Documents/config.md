**File-level: what goes in, what comes out**
Nothing goes "in" — this is a static config module. What comes out is a single `SLMConfig` dataclass instance that every other module (`model.py`, `driver.py`, and eventually the training script) reads hyperparameters from. Not a class you interact with beyond construction; it's a config object, not a computation.

---

**`SLMConfig` (dataclass) — fields**
- `vocab_size` — 1000 placeholder (your real BBPE tokenizer is 8000 — this file's default is stale relative to that)
- `hidden_size` — 384 placeholder
- `num_layers` — 10 placeholder
- `num_heads` — 8 placeholder
- `num_kv_groups` — 2 placeholder (GQA)
- `ffn_multiplier` — 8/3, comment says "rounded internally" — see flag below, this conflicts with your locked SwiGLU convention
- `max_seq_len` — 512 placeholder
- `rms_eps` — 1e-6
- `bias` — False, global flag threaded into sub-modules
- `init_std` — 0.02, GPT-2/Llama convention, consistent with your locked convention
- `tie_embeddings` — True, shares embedding and lm_head weights
- `dropout` — 0.0 placeholder

---

**`head_dim` (property)**
- Computed, not stored: `hidden_size // num_heads`
- At current placeholders: `384 // 8 = 48`

---

**Flagged for your review**
1. `vocab_size=1000` here doesn't match your actual tokenizer vocab of 8000. Fine if this is just a "get something running" scratch config, but don't forget to swap it before a real training run.
2. `bias: bool` is a config-level flag, and `model.py` passes `bias=cfg.bias` into `QKVProjection(...)`. Your `qkv_projection.md` doc doesn't mention a `bias` constructor argument at all — `W_q`/`W_k`/`W_v` were documented as fixed `nn.Linear(..., bias=False)` with no config knob. Worth confirming whether your real `QKVProjection` class actually accepts `bias`, or whether this call will throw a `TypeError` on an unexpected keyword.
3. `ffn_multiplier`'s docstring comment ("rounded internally") foreshadows a real conflict — see `model.md` flag #2.
