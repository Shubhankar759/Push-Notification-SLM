**File-level: what goes in, what comes out**
Two directions of data flow. On save: live training state (model weights, optimizer state, LR scheduler state, GradScaler state, current step, validation loss, config snapshot) goes in, one `.pt` file comes out (plus an overwritten `checkpoint_latest.pt`). On load: a checkpoint path plus the live objects to restore into go in; those objects are mutated in-place, and `(step, val_loss)` come out as the only return value. `find_latest_checkpoint` and `prune_old_checkpoints` are directory-level utilities with no tensor data flow.

---

**Module-level constants / conventions**
- Step checkpoint filename: `checkpoint_step{N}.pt`
- Rolling pointer filename: `checkpoint_latest.pt` (always overwritten, never pruned)
- `config` is stored as a plain dict (`dataclasses.asdict` if a dataclass is passed in) so it can be compared later without needing the original class importable

---

**`_config_to_dict(config)`** (internal)
- Takes: a config object — either a dataclass instance or something already dict-like
- Returns: plain `dict`
- Exists so `save_checkpoint`/`load_checkpoint` don't care whether the caller passes a dataclass or a dict

---

**`_extract_step(filepath)`** (internal)
- Takes: a checkpoint filepath
- Returns: the integer step number parsed via regex from `checkpoint_step{N}.pt`, or `None` if the filename doesn't match
- Used by `prune_old_checkpoints` to sort/identify files without opening them

---

**`_print_config_diff(saved_config, current_config)`** (internal)
- Takes: two plain dicts
- Prints only the keys whose values differ between them (one line per differing key: `key: saved=X current=Y`)
- Pure side-effect function, no return value; called only from `load_checkpoint` when a mismatch is detected

---

**`_partial_load_model_state(model, saved_state_dict)`** (internal)
- Takes: the live `model` and the `model_state_dict` pulled from a checkpoint whose config didn't match the current run
- For each key in the model's *current* `state_dict()`:
  - If missing from `saved_state_dict` → skipped, left at current (freshly-initialized) value
  - If present but shape differs → skipped, left at current value
  - If present and shape matches → copied in from the checkpoint
- Any keys present in `saved_state_dict` but not in the current model are counted and reported as unused/ignored
- Calls `model.load_state_dict()` once at the end with the merged dict
- Prints a one-line summary (counts) plus the actual key names for each skipped category
- Returns: nothing — mutates `model` in place

---

**`save_checkpoint(model, optimizer, scheduler, scaler, step, val_loss, config, checkpoint_dir, keep_last_n)`**
- Takes: all five live training objects, current step, current val loss, config snapshot, target directory, rolling-window size
- **Creates `checkpoint_dir` if it doesn't exist** (`os.makedirs(..., exist_ok=True)`) — confirmed decision, this function owns directory creation, not the caller
- Builds the state dict (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `step`, `val_loss`, `config`)
- Writes it to `checkpoint_step{step}.pt`, then writes the identical content to `checkpoint_latest.pt`
- Calls `prune_old_checkpoints(checkpoint_dir, keep_last_n)` at the end
- Returns: the path to the step checkpoint just written (for logging)

---

**`load_checkpoint(path, model, optimizer, scheduler, scaler, config)`**
- Takes: a specific checkpoint path, the live objects to restore into, and the config the *current* run intends to use
- Loads the file via `torch.load`, extracts `saved_config` and compares against the current config (converted via `_config_to_dict`)
- **If configs match exactly:** full strict `.load_state_dict()` on model, optimizer, scheduler, and scaler — this is the normal "resume interrupted training" path
- **If configs differ (confirmed behavior — architecture-evolution scenario, not a bug state):**
  - Prints a warning and the specific differing config keys via `_print_config_diff`
  - Calls `_partial_load_model_state` to load only matching layers into the model
  - **Deliberately does NOT touch** optimizer/scheduler/scaler — they stay at whatever fresh state the caller constructed them with, since per-parameter optimizer moment buffers and scheduler position don't have a meaningful "partial" mapping onto a changed architecture
  - This does **not** raise/halt — training is expected to continue
- Returns: `(step, val_loss)` read from the checkpoint in both cases — even on a partial/mismatched load, the step counter still resumes from the checkpoint's recorded step (per your confirmed decision), only the optimizer/scheduler/scaler get reset

---

**`find_latest_checkpoint(checkpoint_dir)`**
- Takes: checkpoint directory
- Returns: path to `checkpoint_latest.pt` if it exists, else `None`
- This is the single call a training script makes at startup to decide fresh-run vs. resume-run

---

**`prune_old_checkpoints(checkpoint_dir, keep_last_n)`**
- Takes: checkpoint directory, how many step-checkpoints to retain
- Lists all `checkpoint_step*.pt` files, extracts step numbers via `_extract_step`
- If total count ≤ `keep_last_n`, does nothing
- Otherwise: keeps the newest `keep_last_n` by step number, **plus** whichever single checkpoint currently has the lowest `val_loss` among all step-checkpoints in the directory (confirmed decision — best-ever is always protected, even outside the window)
- Everything else gets `os.remove()`'d; individual delete failures are caught and warned about, not fatal
- Never touches `checkpoint_latest.pt`
- Returns: nothing
- Called internally by `save_checkpoint`, but also usable standalone

---

**Resolved decisions (previously flagged, now locked)**
1. **Best-checkpoint protection:** confirmed — pruning always keeps the best-val_loss checkpoint even outside the keep_last_n window.
2. **Config mismatch behavior:** confirmed — this is *not* treated as a hard error. It's expected to happen when the architecture evolves between runs (e.g., layers added). On mismatch, `load_checkpoint` does a partial, name+shape-matched load of `model_state_dict` only, skips everything that doesn't line up, and leaves optimizer/scheduler/scaler at fresh state. Step/val_loss still resume from the checkpoint's recorded values either way.
3. **Directory creation:** confirmed — `save_checkpoint` owns this via `os.makedirs(checkpoint_dir, exist_ok=True)`.

---

**New item flagged for your review**
`prune_old_checkpoints` determines "best" by calling `torch.load` on every `checkpoint_step*.pt` file still in the directory (to read out its `val_loss` field) whenever the count exceeds `keep_last_n`. This means pruning cost scales with how many stale checkpoints have piled up, and each load pulls a full model+optimizer+scheduler+scaler state dict into memory just to read one float. This wasn't in the original scope so I didn't add anything unrequested to fix it, but flagging the option: a lightweight sidecar index file (e.g. a small JSON `{step: val_loss}` map updated on every `save_checkpoint` call) would let pruning skip opening the big `.pt` files entirely. Let me know if you want that added.
