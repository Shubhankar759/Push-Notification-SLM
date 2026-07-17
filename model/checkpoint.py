"""
checkpoint.py

Save/load/prune logic for training checkpoints. See checkpoint.md for the
full design doc, function contracts, and flagged decisions.

Confirmed behavior (locked as of this version):
- Pruning keeps the newest `keep_last_n` step-checkpoints AND always protects
  the single best-val_loss checkpoint from deletion, even if it falls outside
  that window.
- On config mismatch during load_checkpoint(), this module does NOT hard-raise.
  It performs a partial load of model_state_dict only (matching name + shape),
  skips everything else, and leaves optimizer/scheduler/scaler at their
  freshly-constructed state. step/val_loss are still returned so the training
  loop can resume its step counter.
- save_checkpoint() auto-creates checkpoint_dir via os.makedirs(exist_ok=True).
"""

import os
import re
import glob
import torch
from dataclasses import asdict, is_dataclass


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _config_to_dict(config):
    """Normalize a config (dataclass instance or already a dict) to a plain dict."""
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _extract_step(filepath):
    """Pull the integer step number out of a checkpoint_step{N}.pt filename."""
    match = re.search(r"checkpoint_step(\d+)\.pt$", os.path.basename(filepath))
    return int(match.group(1)) if match else None


def _print_config_diff(saved_config, current_config):
    """Print only the keys that differ between saved and current config."""
    all_keys = set(saved_config.keys()) | set(current_config.keys())
    for key in sorted(all_keys):
        old_val = saved_config.get(key, "<missing>")
        new_val = current_config.get(key, "<missing>")
        if old_val != new_val:
            print(f"[checkpoint]   {key}: saved={old_val}  current={new_val}")


def _partial_load_model_state(model, saved_state_dict):
    """
    Load only the layers of saved_state_dict that exist in model's current
    state_dict AND match shape. Everything else is left at its current
    (freshly-initialized) value. Prints a summary of what was loaded/skipped.
    """
    model_state = model.state_dict()

    loaded_keys = []
    skipped_shape_mismatch = []
    skipped_missing_in_checkpoint = []

    for key, current_tensor in model_state.items():
        if key not in saved_state_dict:
            skipped_missing_in_checkpoint.append(key)
            continue

        saved_tensor = saved_state_dict[key]
        if saved_tensor.shape != current_tensor.shape:
            skipped_shape_mismatch.append(key)
            continue

        model_state[key] = saved_tensor
        loaded_keys.append(key)

    skipped_unused_in_checkpoint = [
        key for key in saved_state_dict.keys() if key not in model_state
    ]

    model.load_state_dict(model_state)

    print(
        f"[checkpoint] Partial load: {len(loaded_keys)} layers loaded, "
        f"{len(skipped_shape_mismatch)} skipped (shape mismatch), "
        f"{len(skipped_missing_in_checkpoint)} skipped (missing in checkpoint), "
        f"{len(skipped_unused_in_checkpoint)} unused checkpoint keys ignored."
    )
    if skipped_shape_mismatch:
        print(f"[checkpoint]   shape-mismatch layers: {skipped_shape_mismatch}")
    if skipped_missing_in_checkpoint:
        print(f"[checkpoint]   missing-in-checkpoint layers (kept fresh init): {skipped_missing_in_checkpoint}")
    if skipped_unused_in_checkpoint:
        print(f"[checkpoint]   unused checkpoint keys (ignored): {skipped_unused_in_checkpoint}")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, step, val_loss, config,
                     checkpoint_dir, keep_last_n):
    """
    Save full training state to checkpoint_step{step}.pt and overwrite
    checkpoint_latest.pt with the same content. Then prunes old checkpoints,
    protecting the best-val_loss checkpoint from deletion.

    Returns: path to the step checkpoint just written.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = _config_to_dict(config)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "config": config_dict,
    }

    step_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    torch.save(state, step_path)
    torch.save(state, latest_path)

    prune_old_checkpoints(checkpoint_dir, keep_last_n)

    return step_path


def load_checkpoint(path, model, optimizer, scheduler, scaler, config):
    """
    Load a checkpoint from `path` and restore state into the live objects
    in-place.

    - If the saved config matches the current config exactly: full strict
      load of model, optimizer, scheduler, scaler.
    - If configs differ: partial load of model_state_dict only (matching
      name+shape layers), optimizer/scheduler/scaler are left untouched
      (fresh state). A warning is printed either way when mismatched.

    Returns: (step, val_loss) from the checkpoint, regardless of which path
    was taken, so the training loop can resume its step counter.
    """
    checkpoint = torch.load(path, map_location="cpu")

    saved_config = checkpoint.get("config", {})
    current_config = _config_to_dict(config)

    if saved_config == current_config:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        print(f"[checkpoint] WARNING: config mismatch detected when loading {path}.")
        _print_config_diff(saved_config, current_config)
        print(
            "[checkpoint] Falling back to partial model-weight load. "
            "Optimizer/scheduler/scaler state will NOT be restored (starting fresh)."
        )
        _partial_load_model_state(model, checkpoint["model_state_dict"])
        # optimizer, scheduler, scaler intentionally left at their
        # freshly-constructed state — not touched here.

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", None)

    return step, val_loss


def find_latest_checkpoint(checkpoint_dir):
    """
    Returns the path to checkpoint_latest.pt if it exists in checkpoint_dir,
    else None. Callers use this to decide fresh-run vs resume-run.
    """
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if os.path.isfile(latest_path):
        return latest_path
    return None


def prune_old_checkpoints(checkpoint_dir, keep_last_n):
    """
    Deletes checkpoint_step*.pt files beyond the newest `keep_last_n`,
    EXCEPT the single checkpoint with the lowest val_loss seen among all
    step-checkpoints currently in the directory, which is always protected
    from deletion regardless of its step position.

    Note: to determine which checkpoint is "best", this function loads each
    checkpoint_step*.pt file's val_loss field (torch.load per file). This is
    only done when there are more files than keep_last_n, but it does mean
    pruning cost grows with however many stale checkpoints have accumulated.
    Flagging this as a known cost — see checkpoint.md.

    checkpoint_latest.pt is never touched by this function.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_step*.pt")
    files = glob.glob(pattern)

    checkpoints = []
    for f in files:
        step = _extract_step(f)
        if step is None:
            continue
        val_loss = None
        try:
            data = torch.load(f, map_location="cpu")
            val_loss = data.get("val_loss", None)
        except Exception:
            pass
        checkpoints.append((step, val_loss, f))

    if len(checkpoints) <= keep_last_n:
        return

    checkpoints.sort(key=lambda entry: entry[0])  # ascending by step

    scored = [c for c in checkpoints if c[1] is not None]
    best_file = min(scored, key=lambda entry: entry[1])[2] if scored else None

    to_keep = {f for _, _, f in checkpoints[-keep_last_n:]}
    if best_file is not None:
        to_keep.add(best_file)

    for step, val_loss, f in checkpoints:
        if f not in to_keep:
            try:
                os.remove(f)
            except OSError as e:
                print(f"[checkpoint] WARNING: failed to prune {f}: {e}")
