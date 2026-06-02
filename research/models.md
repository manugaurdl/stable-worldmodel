# Models — source of truth

One section per trained model: spec, checkpoint, origin, current best-known result per
eval ID, and reproduce command. **This is authoritative.** When a number changes, update
its cell here *and* append a dated entry to [`reproduction-log.md`](reproduction-log.md)
(cite it). Result notation: `<value> [model=<id>@<ckpt>, eval=<eval-id>, seed=<n>]`.
Eval IDs are defined in [`reference.md`](reference.md).

Status legend: ✅ re-verified here · 📄 from result file, not re-verified · 📝 from
progress doc, not verified here.

---

## `quentinll/lewm-pusht`

- **Type:** LeWM — latent world model (ViT-tiny encoder + transformer predictor + action
  embedder, SIGReg loss). Baseline / groundwork.
- **Origin:** pretrained checkpoint downloaded from HuggingFace (`quentinll/lewm-pusht`,
  72 MB). **Not trained here.**
- **Checkpoint:** `$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht/weights.pt`
  (+ `config.json`). ViT keys were remapped old→new HF ViT names to load strict
  (0 missing / 0 unexpected); original backed up to `$STABLEWM_HOME/_backups/`
  (CLAUDE.md trap #5; mapping in `progress/repro_pushT.md`).
- **wandb run ID:** none (predates / external).

**Results**

| eval ID | checkpoint | seed | dataset | success_rate | status |
|---|---|---|---|---|---|
| `pusht-wm-cem` | `weights.pt` | 42 | `pusht_expert_train.h5` | **92.0%** (46/50) | 📄 |
| `pusht-wm-cem` | `weights.pt` | 42 | `pusht_expert_train.lance` | **98%** (49/50) | 📝 |

`92.0% [model=quentinll/lewm-pusht@weights.pt, eval=pusht-wm-cem, seed=42]` — published
LeWM number is ~96%; both runs bracket it. The Lance 98% is reported in
`progress/repro_pushT.md` and is not backed by an on-disk result file on this host.

**Reproduce:** `scripts/repro/lewm_pusht_eval.sh` (≈ 57 s).

---

## `pusht_dinov2_small_psmall`

- **Type:** PreJEPA — DINO-WM reproduction. Frozen `dinov2_small` backbone +
  `CausalPredictor` (depth 6, heads 16); `proprio`/`action` embedded (dim 384+10+10=404),
  per-patch latent prediction. Baseline to build the novel objective on.
- **Origin:** trained here, 2026-05-27, 10 epochs (~25 min/epoch, DDP). Command:
  ```bash
  python scripts/train/prejepa.py dataset_name=pusht_expert_train.h5 \
      subdir=pusht_dinov2_repro num_workers=8
  ```
- **Checkpoint:** `$STABLEWM_HOME/checkpoints/pusht_dinov2_small_psmall/weights_epoch_{1,5,10}.pt`
  (+ `config.json`). **Last = `weights_epoch_10`**; no best/last split.
  ⚠️ Full train `config.yaml` is in the sibling dir `pusht_dinov2_repro/` — same run
  (CLAUDE.md trap #2).
- **Config:** `history_size=3`, `num_preds=1`, `frameskip=5`, `batch_size=32`,
  `lr=5e-4`, `precision=16-mixed`, `max_epochs=10`.
- **wandb run ID:** none (wandb disabled — trap #1; once enabled, run ID would be
  `pusht_dinov2_repro`).

**Results**

| eval ID | checkpoint | seed | dataset | success_rate | status |
|---|---|---|---|---|---|
| `pusht-wm-cem` | `weights_epoch_10` | 42 | `pusht_expert_train.h5` | **48.0%** (24/50) | 📄 |

`48.0% [model=pusht_dinov2_small_psmall@weights_epoch_10, eval=pusht-wm-cem, seed=42]` —
only 10 epochs of training. ⚠️ This eval took **≈ 8728 s (2.4 h)** (trap #3); budget
before re-running.

**Reproduce:** `scripts/repro/prejepa_pusht_eval.sh` (≈ 2.4 h).

---

## `pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep`

- **Type:** PreJEPA — DINO-WM reproduction, **100-epoch** run with the lr-corrected recipe
  (predictor `lr=5e-5`, the paper's value; the 10-epoch `pusht_dinov2_small_psmall` used
  repo default `lr=5e-4`). Same architecture: frozen `dinov2_small` backbone +
  `CausalPredictor` (depth 6, heads 16), per-patch latent prediction, dim 384+10+10=404.
- **Origin:** trained here, started 2026-05-29, 7-GPU DDP (GPUs 1–7, global batch 224),
  100 epochs planned, ckpt every 5 epochs. Training still in progress at eval time
  (epoch 95 on disk). Command: `scripts/repro/prejepa_pusht_train_100ep.sh`.
- **Checkpoint:** `$STABLEWM_HOME/checkpoints/pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep/weights_epoch_{5..95}.pt`
  (+ `config.json`). Evaluated **`weights_epoch_90`** (not the final epoch-100). No best/last split.
  ⚠️ Full train `config.yaml` is in sibling dir `dinowm_pusht_repro_100ep/` (trap #2).
- **Config:** `history_size=3`, `num_preds=1`, `frameskip=5`, `batch_size=32/rank`,
  **`lr=5e-5`**, `precision=16-mixed`, `max_epochs=100`.
- **wandb run ID:** `dinowm_pusht_repro_100ep` (project `manugaur/stable-wm`) — first run with
  a real wandb ID. https://wandb.ai/manugaur/stable-wm/runs/dinowm_pusht_repro_100ep

**Results**

| eval ID | checkpoint | seed | dataset | success_rate | status |
|---|---|---|---|---|---|
| `pusht-wm-cem` | `weights_epoch_90` | 42 | `pusht_expert_train.h5` | **8.0%** (4/50) | ✅ |

`8.0% [model=pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep@weights_epoch_90, eval=pusht-wm-cem, seed=42]`
— ⚠️ **ANOMALY:** far *below* the paper's 92% **and** below this repo's own 10-epoch run
(48%, lr=5e-4). The lr correction (5e-5) was meant to *improve* on 48%; instead it regressed
to 8%. Do not trust as the DINO-WM headline — investigate the wandb `val/mse` curve and
consider evaluating other checkpoints before drawing conclusions. Eval was re-run fresh by us
(✅); runtime ≈ 3.0 h (trap #3).

**Reproduce:** `scripts/repro/dinowm_100ep_pusht_eval.sh` (≈ 3.0 h).

---

## `dinowm_orig_pusht_f5h3`

- **Type:** **Original DINO-WM** (gaoyuezhou/dino_wm `train.py`, *not* our `prejepa.py`) —
  frozen DINOv2 ViT-S/14 (**196** patches; DINO-WM resizes 224→196 pre-backbone, vs PreJEPA's
  256) + `ViTPredictor` (depth 6 / heads 16 / mlp 2048, dim 384+10+10=404) + Conv1d
  proprio/action embedders. `has_decoder=False` (paper's 0.92 "w/o decoder" config; the
  decoder loss is detached so it never affects the predictor). Purpose: a faithful reference
  to debug the PreJEPA repro ("their code vs ours, identical data").
- **Origin:** trained here, **started 2026-06-01**, 8-GPU `accelerate` DDP, global batch 32,
  predictor lr 5e-4, 100 epochs (~50 min/epoch). Trains on our `pusht_expert_train.h5`
  converted to DINO-WM's `pusht_noise` layout, via **precomputed frozen DINOv2 features**
  (bit-identical to on-the-fly; in /dev/shm). Commands:
  `scripts/repro/dinowm_orig_pusht_{dataprep,train}.sh`. ⚠️ Training IN PROGRESS.
- **Checkpoint:** `$STABLEWM_HOME/dino_wm_runs/outputs/dinowm_orig_pusht_f5h3/checkpoints/model_{N}.pth`
  (+ `hydra.yaml` train cfg alongside). Each `model_{N}.pth` is a dict of module objects
  (`predictor`, `action_encoder`, `proprio_encoder`; encoder rebuilt from hub, decoder=None)
  — **not** `load_pretrained`-compatible; eval with the native `pusht-dinowm-plan`.
- **wandb run:** `manugaur/stable-wm` id `lvd4emdj` (name `dinowm_orig_pusht_f5h3_pusht_f5_h3_p1`).
  https://wandb.ai/manugaur/stable-wm/runs/lvd4emdj

**Results**

| eval ID | checkpoint | seed | dataset | success_rate | status |
|---|---|---|---|---|---|
| `pusht-dinowm-plan` | `model_10` | 99 | (val goals) | **32.0%** (16/50) | ✅ |
| `pusht-dinowm-plan` | `model_100` | 99 | (val goals) | _pending (training in progress)_ | 🚧 |

`32.0% [model=dinowm_orig_pusht_f5h3@model_10, eval=pusht-dinowm-plan, seed=99]` —
**interim** (epoch 10 of 100; eval ≈4 h, contended with training on GPU 0). Note this is
DINO-WM's *own* code+eval on our **expert** data; the paper's ~0.92 is at **epoch 100**, so
the headline comparison needs `model_100`. Early read: even the original code is far from
0.92 at 10 ep — consistent with the *data* (expert vs DINO-WM's `pusht_noise`) being a major
factor, but the epoch-100 number is needed to conclude. Val loss still ↓ (0.059→0.044 by ep10).
Plan: run training to 100 uninterrupted (intermediate evals slow it ~3×), then eval the
checkpoint curve (10/25/50/75/100) in parallel on the freed GPUs.
Cross-eval under `pusht-wm-cem` (weight-port → swm `PreJEPA`, reconfigured to 196 patches)
is a separate planned result.

**Reproduce:** `scripts/repro/dinowm_orig_pusht_plan_eval.sh` (EPOCH=N NEVALS=50).

---

## Not yet trained

No models exist for `pldm`, `tdmpc2`, `gcrl` (gcbc/gciql/gcivl/hilp), or for any non-PushT
env (cube / reacher / tworoom). Add a section here when one is trained + evaluated.
