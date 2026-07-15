"""
train.py
========
Full training script for the SLM defined in model.py / config.py.

Run (from the parent directory of this package, since this file uses the
same relative-import style as driver.py):

    python -m your_package_name.train --data_dir ./data --out_dir ./checkpoints

*** DATA ASSUMPTION — READ THIS FIRST ***
None of the uploaded files include a dataset or tokenizer. This script
therefore assumes the standard "nanoGPT-style" data pipeline:

    data_dir/
        train.bin   <- all training token IDs, flat, dtype=np.uint16
        val.bin     <- all validation token IDs, flat, dtype=np.uint16

i.e. you've already tokenized your corpus and dumped the token ID stream
to two flat binary files with something like:

    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    ids.tofile("train.bin")

If your real pipeline differs (e.g. you have a list of pre-tokenized
sequences, a HuggingFace `datasets` object, or a custom Dataset class),
tell me and I'll swap out `get_batch()` / the data-loading section only —
nothing else in this file depends on the data format.

What this script does:
    1. Builds SLM + optimizer (+ optionally resumes from a checkpoint)
    2. Runs the training loop with:
         - linear warmup + cosine decay LR schedule
         - gradient accumulation (for large effective batch size on small GPUs)
         - gradient clipping
         - mixed precision (bf16/fp16 autocast) when on CUDA
    3. Periodically evaluates on val.bin and logs both losses
    4. Saves the best checkpoint (by val loss) and a rolling "latest" checkpoint
       so you can resume after a crash / preemption
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from .config import SLMConfig
from .model import SLM


# --------------------------------------------------------------------------
# Data loading (see assumption above)
# --------------------------------------------------------------------------

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "validation.bin")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Expected '{train_path}' and '{val_path}'. "
            "This script expects pre-tokenized flat uint16 token-ID files. "
            "See the module docstring at the top of train.py."
        )
    # memmap so the whole file doesn't have to fit in RAM
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(data, batch_size, seq_len, device):
    """
    Sample `batch_size` random windows of length `seq_len` from a flat
    token-ID array, returning next-token-prediction (x, y) pairs where
    y is x shifted left by one position.
    """
    max_start = len(data) - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1: i + 1 + seq_len].astype(np.int64)) for i in ix]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay -> min_lr floor
# --------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, data, batch_size, seq_len, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the SLM")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing train.bin and validation.bin")
    p.add_argument("--out_dir", type=str, default="./checkpoints",
                   help="Where to save checkpoints")

    # Optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches for a larger effective batch size")
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging / eval / checkpointing cadence
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)

    p.add_argument("--resume", action="store_true",
                   help="Resume from out_dir/latest.pt if it exists")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    print(f"Using device: {device} (amp={use_amp}, dtype={amp_dtype if use_amp else 'fp32'})")

    # ---- Data ----
    train_data, val_data = load_data(args.data_dir)

    # ---- Model ----
    cfg = SLMConfig()
    model = SLM(cfg).to(device)
    print(f"Model params: {model.num_params() / 1e6:.2f}M total, "
          f"{model.num_params(exclude_embeddings=True) / 1e6:.2f}M non-embedding")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_step = 0
    best_val_loss = float("inf")

    latest_path = os.path.join(args.out_dir, "latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step} (best_val_loss={best_val_loss:.4f})")

    # ---- Training loop ----
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, cfg.max_seq_len, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += step_loss

        # ---- Logging ----
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_loss = running_loss / args.log_every if step > start_step else step_loss
            tokens_per_step = args.batch_size * cfg.max_seq_len * args.grad_accum_steps
            toks_per_sec = tokens_per_step * args.log_every / max(dt, 1e-9)
            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                  f"{toks_per_sec:,.0f} tok/s | {dt:.1f}s")
            running_loss = 0.0
            t0 = time.time()

        # ---- Evaluation ----
        if step % args.eval_every == 0 and step > start_step:
            val_loss = estimate_loss(model, val_data, args.batch_size, cfg.max_seq_len,
                                      device, args.eval_iters)
            print(f"  -> eval: val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                }, os.path.join(args.out_dir, "best.pt"))
                print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

        # ---- Periodic "latest" checkpoint for resuming ----
        if step % args.save_every == 0 and step > start_step:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "config": cfg,
            }, latest_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
