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

- **Trap #7 — host-specific paths live in ONE place: `scripts/hosts/<host>.sh`.** This
  repo was authored on host `manu` (`/home/manu`, `/nas/manu`, conda) and is now also run
  on the CMU `trinity` nodes (`/home/mgaur`, `/data3/mgaur`, `.venv`). Every run script
  **`source`s `scripts/env.sh`**, which loads the right per-host config (selected by
  `SWM_HOST`, default `hostname -s`) and exports `STABLEWM_HOME`, `PY`, `DINO`,
  `NGPU`/`GPUS`, `DINOENV`, `SDL_VIDEODRIVER`, `MUJOCO_GL`. **Adding a machine = add one
  `scripts/hosts/<name>.sh`** (CMU compute nodes just `source` `trinity.sh` and override
  GPUs — same NFS fs). If you find a `/home/manu`, `/nas/manu`, or any absolute
  home/nas/conda path **outside `scripts/hosts/`**, that's the bug — move it into the host
  config, don't paper over it with a one-off env override. `env.sh` fails loud if the
  cache root or interpreter is missing, so a wrong-host selection stops immediately.
  Host selection priority is `SWM_HOST` env var → `~/.swm_host` marker file → `hostname
  -s`. The marker exists because the **two lambda boxes both report `hostname -s` =
  `lambda-scalar`** and differ only by data root: `lambda-75` (`/nas/manu`, = the `manu`
  config) vs `lambda-74` (`/data_new/manu`). Drop a one-liner per box —
  `echo lambda-75 > ~/.swm_host` — and a bare `bash scripts/repro/<x>.sh` resolves
  correctly there.

- **Trap #8 — DINO-WM's DINOv2 hub load is unpinned + the `dino_wm` conda env is Python
  3.9.** `dino_wm/models/dino.py` does `torch.hub.load("facebookresearch/dinov2", …)`
  with **no ref**, so each machine pulls whatever DINOv2 `main` is current — both a
  cross-node nondeterminism risk *and* a hard break: the latest DINOv2
  `layers/{block,attention}.py` use PEP 604 `X | None` annotations that raise
  `TypeError: unsupported operand type(s) for |` under Python 3.9 (the env built from
  `dino_wm/environment.yaml`). Fix lives **in our code**: `_enable_py39_dinov2()` in
  `dino.py` injects `from __future__ import annotations` into dinov2 modules at import
  (in memory — the torch-hub cache is never edited; no-op on ≥3.10; features unchanged).
  Do **not** patch the `~/.cache/torch/hub` copy in place (out-of-repo dependency).
  *Still TODO:* pin the hub commit for true reproducibility. The `dino_wm` env lives in
  NFS home (`/home/mgaur/miniconda3/envs/dino_wm`, built 2026-06-03) so it is shared
  across CMU nodes; `DINOENV` in `scripts/hosts/trinity.sh` points at it.

- **Trap #9 — `dataprep.sh` precompute hid shard failures (fixed).** The feature-precompute
  loop backgrounded 8 shards then called bare `wait`, which returns 0 regardless of child
  exit codes — so a total precompute failure still reported **exit 0 with 0 features**
  written. Fixed to `wait "$pid"` per shard (fail loud) + assert final counts
  (`16816`/`1869`). If you copy this fan-out pattern elsewhere, wait per-PID.

- **Trap #10 — DINO-WM feats in `/dev/shm` collide with the DataLoader (Bus error).** The
  precompute caches ~295 GB of train feats in tmpfs (`/dev/shm/pusht_feats`) for RAM-speed
  reads, but PyTorch's DataLoader also passes prefetched batches **through `/dev/shm`**
  (both `file_descriptor` and `file_system` sharing strategies do — switching strategy does
  NOT help). On a box where `/dev/shm` is small or shared (e.g. **lambda-hyperplane**: 504 GB
  tmpfs with a neighbor's 200 GB file already in it), 295 GB of feats leaves no room and the
  workers die: `DataLoader worker killed by signal: Bus error … out of shared memory`,
  surfacing as a `ChildFailedError` with exit 1. Worse at high `batch*num_workers`
  (`train.py:151` uses `prefetch_factor=4`). **Fix:** move feats off `/dev/shm` to a fast
  **local** disk via `SWM_FEATS_ROOT` in `scripts/hosts/<host>.sh` (page cache keeps them
  RAM-hot; also reboot-persistent). On lambda-hyperplane that's `/data` (local nvme ext4).
  **Do NOT use NFS** for this: `/data_new` (98% full) moved/wrote 16816 small files at
  ~1 file/sec (serial *and* 32-way parallel) — a 245 GB move would take hours; local `/data`
  did 9000 files/min. (`dataprep.sh` step 2 reads `SWM_FEATS_ROOT`, default
  `/dev/shm/pusht_feats`.)

---

## Conventions

- **Env setup is host-specific — never hardcode it.** `source scripts/env.sh` at the top
  of every run script (after `set -euo pipefail`); it resolves the repo root, loads
  `scripts/hosts/<host>.sh`, and exports `STABLEWM_HOME`, `PY` (interpreter), `DINO`,
  `NGPU`/`GPUS`, `SDL_VIDEODRIVER=dummy`, `MUJOCO_GL=egl`. Pick the machine with
  `SWM_HOST=<name>` (default `hostname -s`), e.g.
  `SWM_HOST=trinity-0-3 bash scripts/repro/<x>.sh`. CMU site:
  `STABLEWM_HOME=/data3/mgaur/stable_worldmodel`, `.venv` interpreter, GPUs on `trinity-0-*`
  compute nodes (login `trinity` has none). Original `manu` host: `/nas/manu/...` + conda.
  See Trap #7 and `research/reference.md`. (Use `CUDA_VISIBLE_DEVICES` per-run to pin a
  single GPU for evals.)
- **Naming:** model IDs as above; eval IDs in `reference.md`; one repro script per
  result/table in `scripts/repro/`.
- Keep `research/` docs in their lanes (orientation / reference / source-of-truth /
  presentation). Don't let numbers leak out of the source-of-truth files.
