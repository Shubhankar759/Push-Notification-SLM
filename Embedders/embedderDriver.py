"""
Using TokenEmbedding / LMHead with your train.bin / validation.bin
=====================================================================

Your data pipeline already wrote out:
    train.bin
    validation.bin

as flat binary files of uint16 token IDs (memmapped, per your handoff
doc / project memory). This script shows the full path:

    .bin file on disk
        -> np.memmap (lazy, doesn't load the whole file into RAM)
        -> random batch of (input, target) sequences
        -> torch.LongTensor
        -> TokenEmbedding forward pass
        -> LMHead forward pass (tied weights)
        -> cross-entropy loss against targets

Run this directly: python use_token_embedding.py
Edit the CONFIG block below to point at your real file paths and the
real final vocab_size from your BBPE tokenizer training run.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from token_embedder import TokenEmbedding, LMHead, build_tied_embedding_and_head # in the folders


# ---------------------------------------------------------------------
# CONFIG — edit these to match your actual setup
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_BIN_PATH = PROJECT_ROOT / "train.bin"
VAL_BIN_PATH = PROJECT_ROOT / "validation.bin"

VOCAB_SIZE = 710      # <-- IMPORTANT: set this to the EXACT vocab_size
                        #     your BBPE tokenizer.json reports, not just
                        #     "approximately 8000". A mismatch here is
                        #     the #1 source of index-out-of-range errors.
HIDDEN_SIZE = 512       # placeholder until your full architecture
                        # (num_layers, num_heads, num_kv_groups) is locked

BATCH_SIZE = 8
SEQ_LEN = 128           # how many tokens per training example (context length)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# STEP 1 — load the .bin files as memmaps
# ---------------------------------------------------------------------
def load_bin(path: str) -> np.memmap:
    """
    Loads a uint16 binary token file as a numpy memmap. This does NOT
    read the whole file into RAM — it lazily reads only the slices you
    actually index into, which is why your pipeline used this format
    for million-row scale in the first place.
    """
    return np.memmap(path, dtype=np.uint16, mode="r")


# ---------------------------------------------------------------------
# STEP 2 — sample a random batch of (input, target) sequences
# ---------------------------------------------------------------------
def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: str):
    """
    Standard next-token-prediction batching:
        input  = tokens[i   : i+seq_len]
        target = tokens[i+1 : i+seq_len+1]   (shifted by one position)

    Returns two LongTensors of shape (batch_size, seq_len).
    """
    # Pick random starting positions, leaving room for seq_len+1 tokens
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([data[i     : i + seq_len]     for i in starts])
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in starts])

    # uint16 -> int64 (torch embedding lookup requires long dtype)
    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))

    if device.type == "cuda":
        # pinned memory + non_blocking overlaps the H2D copy with compute
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# ---------------------------------------------------------------------
# STEP 3 — sanity check: make sure no token ID exceeds vocab_size
# ---------------------------------------------------------------------
def check_vocab_bounds(data: np.memmap, vocab_size: int, name: str, sample_size: int = 2_000_000):
    """
    Scans a sample of the data (not necessarily the whole file, for
    speed on large files) and confirms every token ID is < vocab_size.
    A mismatch here is the most common bug when wiring a tokenizer to
    an embedding table — better to catch it now than 500 steps into
    training with a cryptic CUDA assert.
    """
    sample = data[:sample_size] if len(data) > sample_size else data[:]
    max_id = int(sample.max())
    if max_id >= vocab_size:
        raise ValueError(
            f"[{name}] Found token ID {max_id} but vocab_size={vocab_size}. "
            f"Your TokenEmbedding's vocab_size does not match the tokenizer "
            f"that produced this .bin file. Fix VOCAB_SIZE in the CONFIG "
            f"block above to match your tokenizer.json's actual vocab size."
        )
    print(f"[{name}] OK — max token ID in sample = {max_id}, vocab_size = {vocab_size}")


# ---------------------------------------------------------------------
# MAIN — wire it all together
# ---------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")

    # Load data
    train_data = load_bin(TRAIN_BIN_PATH)
    val_data = load_bin(VAL_BIN_PATH)
    print(f"train.bin: {len(train_data):,} tokens")
    print(f"validation.bin: {len(val_data):,} tokens")

    # Sanity check vocab bounds before doing anything else
    check_vocab_bounds(train_data, VOCAB_SIZE, "train")
    check_vocab_bounds(val_data, VOCAB_SIZE, "validation")

    # Build the tied embedding + output head
    token_embedding, lm_head = build_tied_embedding_and_head(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    token_embedding.to(DEVICE)
    lm_head.to(DEVICE)

    # Confirm tying actually worked: same underlying tensor, not a copy
    assert token_embedding.weight is lm_head.token_embedding.weight
    print("Tied weights confirmed: TokenEmbedding and LMHead share the same tensor.")

    # Pull one batch and push it through embedding -> (stand-in for
    # transformer blocks, not built yet) -> lm_head -> loss
    x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
    print(f"\nInput batch shape:  {tuple(x.shape)}   (batch_size, seq_len)")
    print(f"Target batch shape: {tuple(y.shape)}   (batch_size, seq_len)")

    # 1. Token embedding lookup
    embedded = token_embedding(x)
    print(f"\nAfter TokenEmbedding: {tuple(embedded.shape)}   (batch_size, seq_len, hidden_size)")

    # 2. NOTE: in the real model, `embedded` flows through your GQA +
    #    RoPE + SwiGLU transformer blocks here. Those aren't built yet,
    #    so for this demo we feed `embedded` straight into lm_head just
    #    to prove the tied embedding/output path works end to end.
    logits = lm_head(embedded)
    print(f"After LMHead:         {tuple(logits.shape)}   (batch_size, seq_len, vocab_size)")

    # 3. Loss against targets (standard next-token cross-entropy)
    #    logits: (batch_size, seq_len, vocab_size) -> flatten to (N, vocab_size)
    #    targets: (batch_size, seq_len)            -> flatten to (N,)
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        y.view(-1),
    )
    print(f"\nCross-entropy loss (untrained, random init): {loss.item():.4f}")
    print(f"For reference, ln(vocab_size) = {torch.log(torch.tensor(float(VOCAB_SIZE))).item():.4f} "
          f"is roughly what an untrained model's loss should be near.")


if __name__ == "__main__":
    main()