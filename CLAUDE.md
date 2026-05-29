# CLAUDE.md — invariants for this repo

> **First read:** open [`research/README.md`](research/README.md) at the start of every
> session. It is the orientation doc: project goal, workflow, current models, key
> results, and active todos. This file holds only the rules that override default
> behavior — read it, then go there.

This is a research checkout of **`stable-worldmodel`** (upstream `galilai-group`, fork
`manugaur`). The library code is upstream; the *research* — training world models and
evaluating them with MPC on PushT — is ours and is documented under `research/`.

---

## The reproducibility rules (non-negotiable)

1. **A number is never reported alone.** Every result is written as
   `<value> [model=<model-id>@<checkpoint>, eval=<eval-id>, seed=<n>]`. A bare number
   with no IDs is **unverified** and must be labeled as such.

2. **Two IDs identify every result:**
   - **Model ID** — one trained model. *Legacy* models (no wandb run): the checkpoint
     directory name under `$STABLEWM_HOME/checkpoints/` (e.g. `quentinll/lewm-pusht`,
     `pusht_dinov2_small_psmall`), always paired with a checkpoint epoch
     (`@weights_epoch_10`). *Future* models: the **wandb run ID** in project `stable-wm`
     (= `subdir`; wandb logs there as of 2026-05-29 — see trap #1).
   - **Eval ID** — one named evaluation type, pinning the exact command. Defined in
     [`research/reference.md`](research/reference.md) (`pusht-wm-cem`, `pusht-ff`).

3. **All authoritative numbers live in [`research/models.md`](research/models.md) and
   [`research/reproduction-log.md`](research/reproduction-log.md).** Presentation /
   narrative (the `progress/*.md` working journal, slides) cite IDs and copy from there —
   they are never the source.

4. **Every result ships a reproduce command** — copy-pasteable, including env setup
   (`STABLEWM_HOME`, `CUDA_VISIBLE_DEVICES`, `SDL_VIDEODRIVER`, `MUJOCO_GL`). Reproduction
   bash lives in `scripts/repro/`, not inline in prose.

5. **Code changes that affect a number** are documented with the commit hash,
   `file:line` refs, and the verification check that proves correctness.

---

## Before reporting ANY number — checklist

Run this every time, before writing a number anywhere:

- [ ] **Which model ID?** (checkpoint dir name, or wandb run ID once wandb is on.)
- [ ] **Which checkpoint?** Checkpoints here are epoch-numbered (`weights_epoch_{N}.pt`);
      "last" = highest epoch. There is **no** best/last split — no val-based selection
      exists. State the exact file.
- [ ] **Which eval ID?** (`pusht-wm-cem` etc. — the exact command, from `reference.md`.)
- [ ] **Run the eval fresh** — never copy from memory or a stale doc. If you transcribe
      from an existing result file instead of re-running, **label it
      "from result file — not re-verified"**.
- [ ] **Log it** to `research/reproduction-log.md` (dated) and update `research/models.md`.

---

## Mistakes & traps log (institutional memory)

Append here whenever a bug corrupts a number or a structural quirk wastes time. Record
the wrong value, the correct value, the root cause (`file:line`), and the fix.

- **Trap #1 — wandb (now ON by default; was silent before 2026-05-29).** The wandb block
  is injected from `scripts/train/config/launcher/local.yaml`. It now defaults to
  **`wandb.enabled: true`, `entity: manugaur`, `project: stable-wm`** — previously
  `enabled: false` with a placeholder `entity: stable-wm`, so **every run before
  2026-05-29 logged nothing** (legacy model IDs = checkpoint dir names, no wandb run).
  Since `id: ${subdir}`, the wandb run ID equals `subdir` (= run dir name) — **so you must
  pass `subdir=<name>`** or the run ID is empty and checkpoints land in the `checkpoints/`
  root. New runs (prejepa) thus have a wandb run ID aligned with the dir-name convention.
  Disable for throwaway runs with `wandb.enabled=false`. (Verify with
  `wandb.Api().runs("manugaur/stable-wm")`.) Metrics logged by `prejepa.py`:
  `train/mse` + per-modality `train/*_loss` (per step), `val/mse` (per epoch, the proxy we
  watch vs MPC), `train/grad_norm`, `perf/{samples_per_sec,step_time_s}`, and lr.

- **Trap #2 — config and weights live in DIFFERENT dirs.** In `scripts/train/*.py`,
  `subdir` sets the run dir (gets the full `config.yaml`) while `output_model_name` sets
  the checkpoint dir (gets `weights_*.pt` + the model-only `config.json` that
  `load_pretrained` actually uses). For the PreJEPA run these differ:
  `pusht_dinov2_repro/` (config.yaml only, **no weights**) vs
  `pusht_dinov2_small_psmall/` (weights + config.json). **Same run, two dirs** — don't
  treat them as two models. (`scripts/train/prejepa.py:279,298`.)

- **Trap #3 — PreJEPA eval is expensive.** `pusht-wm-cem` on the PreJEPA dinov2_small
  model took **~8728 s (≈2.4 h)** for 50 episodes, vs **~57 s** for LeWM. Budget for it;
  do not casually re-run to "double-check".

- **Trap #4 — transient NFS `KeyError` on first eval.** The first eval right after
  decompressing the h5 can throw `KeyError` in `eval_wm.py` (~line 137, `max_start_idx_dict[ep_id]`)
  from a stale/partial NFS read. **Just re-run** — do not edit the index-lookup code.

- **Trap #5 — LeWM HF weights need a ViT key remap.** The HF checkpoint
  `quentinll/lewm-pusht` was saved with old HuggingFace ViT layer names; the installed
  `transformers` builds the refactored ViT. Names were remapped to load strict (0
  missing / 0 unexpected); original backed up to `$STABLEWM_HOME/_backups/`. See
  `research/reference.md`.

- **Trap #6 — `eval_ff.py` is not Lance-safe.** `eval_ff.py` still uses the pre-fix
  col-name heuristic and `get_row_data(...)['episode_idx'/'step_idx']` that
  `eval_wm.py` was fixed away from (commit `7057f0f`). It will KeyError on Lance
  datasets. Use h5 with `eval_ff.py`, or port the `get_col_data` fix.

---

## Conventions

- **Env setup** for any run (this host = `manu`): `STABLEWM_HOME=/nas/manu/stable_worldmodel`
  (home is ~99% full; `/nas` is NFS), `CUDA_VISIBLE_DEVICES=0`, `SDL_VIDEODRIVER=dummy`,
  `MUJOCO_GL=egl`. Full setup in `research/reference.md`.
- **Naming:** model IDs as above; eval IDs in `reference.md`; one repro script per
  result/table in `scripts/repro/`.
- Keep `research/` docs in their lanes (orientation / reference / source-of-truth /
  presentation). Don't let numbers leak out of the source-of-truth files.
