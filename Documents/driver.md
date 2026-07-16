**File-level: what goes in, what comes out**
Not a module you import — a standalone script (`python driver.py`). Takes nothing as input beyond `config.py`'s defaults; produces no return value, only printed diagnostics to stdout. Nothing is written to disk.

---

**What it does, in order**

1. **Build the model** — constructs `SLMConfig()` and `SLM(cfg)`, seeds `torch.manual_seed(0)` for reproducibility.
2. **Parameter count check** — prints every config field, then `model.num_params()` (total) and `model.num_params(exclude_embeddings=True)`, comparing total against a hard-coded 15M target and flagging if off by more than 15%.
3. **Forward-pass shape check** — builds random dummy token IDs `(batch_size=4, seq_len=32)`, runs `model.eval()` + `torch.no_grad()` forward, asserts output shape is `(4, 32, vocab_size)`.
4. **Dummy training step (gradient flow check)** — switches to `model.train()`, builds an `AdamW` optimizer and `CrossEntropyLoss`, runs one real `backward()` + `optimizer.step()` on random dummy targets, checks every named parameter received a gradient (`p.grad is not None`), and prints the loss alongside the theoretical random-init expectation `ln(vocab_size)`.

---

**Why this script matters right now**
This is effectively steps 1–2 of the "correctness verification plan" you'd outlined earlier (loss-at-init check, gradient flow check) already implemented, just using random/dummy data instead of your real tokenized corpus. It's the right smoke test before spending any real compute — but note it uses `torch.randint` targets, not your actual `train.bin`, so a "PASS" here confirms wiring is correct, not that your data pipeline or model quality is correct.

---

**Flagged for your review**
1. The loss-at-init sanity check here is exactly right in spirit — `ln(vocab_size)` — but with `vocab_size=1000` (current config default) that target is `ln(1000)≈6.91`, not the `ln(8000)≈8.99` you'd want once `config.py`'s `vocab_size` is corrected to match your real 8000-token BBPE tokenizer.
2. No mixed precision, no gradient accumulation, no gradient clipping, no LR schedule here — appropriately so, since this script's only job is "does everything wire together and receive gradients," not "is this a real training loop." Those all still need to be built separately.
