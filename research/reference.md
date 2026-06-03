# Static reference

Rarely-changing facts: env setup, storage paths, datasets, **eval IDs (exact
commands)**, training templates, naming, model internals, CLI, gotchas. Numbers do **not**
live here — see [`models.md`](models.md) / [`reproduction-log.md`](reproduction-log.md).

---

## 1. Environment setup (cross-node — see CLAUDE.md Trap #7)

Host-specific facts (cache root, interpreter, GPUs) live in `scripts/hosts/<host>.sh`,
loaded by `scripts/env.sh`. **Don't hardcode host paths** — source the loader instead:

```bash
# every run script does this; for an ad-hoc shell, do it yourself:
source scripts/env.sh                 # picks scripts/hosts/$SWM_HOST.sh (default: hostname -s)
# now exported: STABLEWM_HOME, PY, DINO, NGPU/GPUS, DINOENV, SDL_VIDEODRIVER, MUJOCO_GL
# pick a machine explicitly:  SWM_HOST=trinity-0-3 source scripts/env.sh
# pin a single GPU for evals:  export CUDA_VISIBLE_DEVICES=0
```

Per-host configs currently defined:

| host | `STABLEWM_HOME` | interpreter (`PY`) | GPUs | DINO-WM conda env (`DINOENV`) |
|---|---|---|---|---|
| `trinity` (CMU login) | `/data3/mgaur/stable_worldmodel` (NFS) | repo `.venv` (uv py3.10) | none — ssh to a node | `/home/mgaur/miniconda3/envs/dino_wm` (py3.9) |
| `trinity-0-3` (CMU node) | ↑ inherited | ↑ inherited | 8× RTX 6000 Ada 48 GB | ↑ inherited (shared NFS home) |
| `manu` (original) | `/nas/manu/stable_worldmodel` | repo `.venv` | 8× RTX 49 GB | `/nas/manu/miniconda3/envs/dino_wm` |

One-time per-host venv build:

```bash
uv venv --python=3.10 && source .venv/bin/activate
uv sync --extra all --group dev
uv pip install scikit-learn datasets                 # missing from every extra
uv pip install --reinstall "pygame==2.6.1"           # stock build crashes on import (missing FRect)
```

`STABLEWM_HOME` is the cache root (`get_cache_dir`); defaults to `~/.stable_worldmodel`
if unset. **Always set it** (via `env.sh`) — datasets + checkpoints are large.

## 2. Storage layout — `$STABLEWM_HOME/`

```
datasets/
  pusht_expert_train.h5          # 46 GB decompressed; the PushT expert data
  pusht_expert_train.h5.zst      # 13 GB compressed source
  (pusht_expert_train.lance)     # optional Lance conversion (~5.6 GB)
checkpoints/
  models--quentinll--lewm-pusht/ # LeWM HF checkpoint: weights.pt + config.json
  pusht_dinov2_small_psmall/     # PreJEPA weights_epoch_{1,5,10}.pt + config.json + pusht_results.txt
  pusht_dinov2_repro/            # PreJEPA full train config.yaml ONLY (same run — CLAUDE.md trap #2)
  quentinll/                     # LeWM eval outputs: env_*.mp4 + pusht_results.txt
  <run>/env_*.mp4, pusht_results.txt   # eval writes here, next to the checkpoint
_backups/                        # original (pre-remap) LeWM weights.pt
```

## 3. Datasets

`pusht_expert_train` — expert PushT demos. 18,685 episodes, 2,336,736 steps
(~125 steps/ep, range 49–246). Per timestep:

| Column | Dim | Meaning |
|---|---|---|
| `pixels` | 224×224 RGB | rendered frame (PNG bytes) |
| `state` | 7 | agent x,y · block x,y · block angle · agent vx,vy |
| `proprio` | 4 | agent x,y,vx,vy |
| `action` | 2 | expert commanded move |

Loader autodetects format (h5 / Lance / folder / video). In eval the dataset only
supplies **start states and goal states** (goal = start + 25 steps); the pixels the model
sees are rendered live from the env. Inspect with `swm inspect pusht_expert_train`.
Convert with `swm convert pusht_expert_train pusht_expert_train.lance --dest-format lance`.

## 4. Eval IDs (the exact commands)

A result must name one of these. Both use Hydra config `scripts/plan/config/pusht.yaml`.

### `pusht-wm-cem` — world-model MPC eval (the main one)

`scripts/plan/eval_wm.py`. Loads a world model via `load_pretrained`, wraps it in a CEM
solver + `WorldModelPolicy`, runs receding-horizon MPC in 50 parallel `swm/PushT-v1`
envs. The real simulator decides success.

```bash
# (env exports from §1 first)
python scripts/plan/eval_wm.py policy=<model-id> eval.dataset_name=pusht_expert_train.h5
```
- LeWM:    `policy=quentinll/lewm-pusht`
- PreJEPA: `policy=pusht_dinov2_small_psmall/weights_epoch_10.pt`

**Pinned defaults** (`config/pusht.yaml` + `config/solver/cem.yaml`):

| Knob | Value | Knob | Value |
|---|---|---|---|
| `seed` | 42 | CEM `num_samples` | 300 |
| `eval.num_eval` | 50 | CEM `n_steps` (iters) | 30 |
| `eval.eval_budget` | 50 | CEM `topk` | 30 |
| `eval.goal_offset_steps` | 25 | CEM `var_scale` | 1.0 |
| `plan_config.horizon` | 5 | `plan_config.receding_horizon` | 5 |
| `plan_config.action_block` | 5 | `eval.img_size` | 224 |

**Success criterion:** block within 20 px and 20° of the goal pose.
**Output:** `success_rate` (% of 50) → printed + appended to
`$STABLEWM_HOME/checkpoints/<policy-parent>/pusht_results.txt`, plus 50 `env_*.mp4`.
**Cost:** LeWM ≈ 57 s; PreJEPA dinov2_small ≈ 2.4 h (trap #3).
**Variants:** add `eval.dataset_name=pusht_expert_train.lance` for Lance; `bf16=true`,
`compile=true` for speed. Each variant is a *different* result — note it.

### `pusht-dinowm-plan` — DINO-WM's *native* planning eval (for `dinowm_orig_*` models)

The **original** DINO-WM repo's own evaluator: `dino_wm/plan.py` + `dino_wm/conf/plan_pusht.yaml`.
Used only for models trained with the original code (e.g. `dinowm_orig_pusht_f5h3`), which
are not `load_pretrained`-compatible. Runs in the `dino_wm` conda env (py3.9), not the swm venv.

```bash
source scripts/env.sh                  # exports STABLEWM_HOME, DINO, DINOENV (needs a host with DINOENV set)
export DATASET_DIR="$STABLEWM_HOME/datasets" WANDB_MODE=disabled
cd "$DINO" && CUDA_VISIBLE_DEVICES=<gpu> "$DINOENV" plan.py --config-name plan_pusht.yaml \
    ckpt_base_path="$STABLEWM_HOME/dino_wm_runs" \
    model_name=dinowm_orig_pusht_f5h3 model_epoch=<N> n_evals=50 n_plot_samples=0
# wrapper: scripts/repro/dinowm_orig_pusht_plan_eval.sh   (EPOCH=<N> NEVALS=50 GPU=<g>)
```

**Pinned defaults** (`plan_pusht.yaml`): `seed=99`, `n_evals=50`, MPC-CEM
(`num_samples=300`, `opt_steps=30`, `topk=30`, `horizon=5`, `n_taken_actions=5`,
`max_iter=null`), `goal_source='dset'`, `goal_H=5`. `n_plot_samples=0` (no decoder viz;
we train `has_decoder=False`).

**Goal/success:** goal = a val-trajectory state `goal_H*frameskip = 25` env-steps ahead;
success = block pos `<20 px` **and** angle `<π/9` at the final step (`env.eval_state`).
**Output:** `mpc/success_rate` in `dino_wm/plan_outputs/<ts>_dinowm_orig_pusht_f5h3_gH5/logs.json`.

> **Not the same as `pusht-wm-cem`.** Same CEM knobs and 25-step goal distance, but a
> *different evaluator + sim + seed* (99 vs 42) and DINO-WM's own `eval_state`. This is the
> protocol behind DINO-WM's published ~0.92. Cross-eval under `pusht-wm-cem` is a *separate*
> result (requires porting the checkpoint into a swm `PreJEPA` — see `models.md`).

### `pusht-ff` — feed-forward / BC policy eval

`scripts/plan/eval_ff.py`. For policies that emit actions directly (`AutoActionableModel`
+ `FeedForwardPolicy`), no planning. ⚠️ **Lance-unsafe** (CLAUDE.md trap #6): use an h5
dataset, or port the `get_col_data` fix from `eval_wm.py`. No model trained for this yet.

## 5. Training templates

Hydra scripts in `scripts/train/`. Checkpoints save to
`$STABLEWM_HOME/checkpoints/${output_model_name}/weights_epoch_{N}.pt` + `config.json`;
the full train `config.yaml` saves to `…/${subdir}/` (different dir — trap #2).

### PreJEPA (DINO-WM reproduction) — `scripts/train/prejepa.py` + `config/prejepa.yaml`

As actually run (the `pusht_dinov2_small_psmall` model):
```bash
# (env exports from §1; training uses DDP across all visible GPUs — devices=auto)
python scripts/train/prejepa.py \
    dataset_name=pusht_expert_train.h5 \
    subdir=pusht_dinov2_repro \
    num_workers=8
```
Key knobs (`config/prejepa.yaml`): `backbone.name=dinov2_small` (frozen), predictor
`CausalPredictor` depth 6 / heads 16, `wm.history_size=3`, `wm.num_preds=1`,
`frameskip=5`, `batch_size=32`, `optimizer.lr=5e-4`, `trainer.max_epochs=10`,
`precision=16-mixed`. `output_model_name=pusht_${backbone.type}_p${predictor.size}`
→ `pusht_dinov2_small_psmall`. **Checkpoint cadence:** `SaveCkptCallback(epoch_interval=5)`
→ saves epochs 5, 10 (+ final). ~25 min/epoch on this host.

### LeWM — `scripts/train/lewm.py` + `config/lewm.yaml`

Not trained here (we use the pretrained HF checkpoint). Template: vit `tiny` encoder
(`embed_dim=192`), `Predictor` depth 6, `SIGReg` loss (weight 0.09), `max_epochs=100`,
`lr=5e-5`, `batch_size=128`. `output_model_name=lewm`, `subdir=${hydra:job.id}`.

### Enabling wandb (currently OFF — see trap #1)

wandb config is injected from `scripts/train/config/launcher/local.yaml` (`@package
_global_`), with **`wandb.enabled: false`** by default and `entity: stable-wm` (a
placeholder — your entity is `manugaur`). `id: ${subdir}`, so once enabled the **wandb run
ID equals the `subdir`** (and thus the run dir name). To turn on:
```bash
python scripts/train/prejepa.py ... wandb.enabled=true wandb.config.entity=manugaur
```
Then a future model's ID is its wandb run ID (= subdir). Confirm with
`wandb.Api().runs("manugaur/stable-wm")`.

## 6. Model IDs & checkpoints

- **Resolution** (`load_pretrained`, `stable_worldmodel/wm/utils.py`): a `.pt` path, a
  folder with exactly one `.pt` + `config.json`, or a HF `<user>/<repo>` id. All relative
  to `$STABLEWM_HOME/checkpoints/`. `instantiate(config.json)` + `load_state_dict`.
- **Model ID** = checkpoint dir name (legacy) or wandb run ID (future). Always pair with
  the checkpoint epoch: `pusht_dinov2_small_psmall@weights_epoch_10`.
- **No best/last split.** Checkpoints are epoch-numbered; "last" = highest epoch. No
  val-based selection exists.

## 7. Model internals (where to look)

Eval call path (both models):
```
eval_wm.py → load_pretrained(<id>) → World("swm/PushT-v1", num_envs=50).evaluate(...)
  per step: WorldModelPolicy.get_action → CEMSolver.solve → model.get_cost(obs, actions)
            (encode goal, rollout latents, MSE-to-goal cost) → success = env terminated
```
- **LeWM**: `stable_worldmodel/wm/lewm/lewm.py`, `module.py`. ViT encoder + transformer
  `Predictor` + action `Embedder`; trained with prediction + `SIGReg` loss (`wm/loss.py`).
- **PreJEPA** (DINO-WM): `stable_worldmodel/wm/prejepa/prejepa.py`, `module.py`. Frozen
  DINOv2 backbone + `CausalPredictor`; per-patch latent prediction. `proprio`/`action`
  encoded by small `Embedder`s and concatenated to patch embeddings (dim 384+10+10=404).
- **Solvers**: `stable_worldmodel/solver/` (`cem.py` default; also mppi, icem, gd, pgd).
- **MPC wrapper**: `stable_worldmodel/policy.py` (`WorldModelPolicy`, `PlanConfig`).

## 8. CLI quick reference (`swm`)

| Command | Use |
|---|---|
| `swm datasets` | list datasets in `$STABLEWM_HOME/datasets` |
| `swm inspect <name>` | columns, shapes, #episodes/steps |
| `swm checkpoints [filter]` | list checkpoints grouped by run |
| `swm convert <name> [out] -f lance` | convert dataset format |
| `swm envs` / `swm fovs <env>` | list envs / factors of variation |

## 9. Gotchas

Install/runtime: missing deps (`scikit-learn`, `datasets`); broken pygame (reinstall
2.6.1); headless (`SDL_VIDEODRIVER=dummy`); HF downloaders use `urllib` and fail SSL here
→ fetch with `curl`. The reproducibility-critical traps (wandb silent, config/weights
dir-split, 2.4 h PreJEPA eval, transient NFS KeyError, LeWM ViT remap, `eval_ff.py`
Lance-unsafe) are in [`../CLAUDE.md`](../CLAUDE.md).

## 10. Documentation lanes

| Where | Role | Mutability |
|---|---|---|
| `../CLAUDE.md` | invariants, checklist, traps | append traps |
| `README.md` | orientation (read first) | living |
| `reference.md` (this) | static facts + exact commands | rare edits |
| `models.md` / `reproduction-log.md` | **authoritative numbers** | append-only |
| `../progress/*.md` | **working journal + presentation** — what I'm doing, why, narrative + cited results per effort (e.g. `repro_pushT.md`) | living per-effort |
| `../scripts/repro/*.sh` | one reproduce script per result | per result |

**Lane rule for `progress/` vs source-of-truth:** the progress journal narrates the work
and may quote results, but every number it states **cites its model + eval ID** and must
match `models.md` / `reproduction-log.md`, which remain the canonical record. Story lives
in `progress/`; the ledger lives in `research/`.
