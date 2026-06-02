# Reproduction log — source of truth (append-only)

Chronological ledger of evaluation/training runs actually executed. **Append only —
never edit a past entry.** A wrong number stays; a later entry corrects it. Every result
cites `model=<id>@<ckpt>, eval=<eval-id>, seed=<n>`. When an entry produces a number,
also update the cell in [`models.md`](models.md).

Status: ✅ re-verified · 📄 from on-disk result file, not re-run by us · 📝 reported in a
progress doc, not on-disk here.

Entry template:
```
### YYYY-MM-DD — <short title>
- model: <id>@<ckpt>   eval: <eval-id>   seed: <n>   dataset: <name>
- result: <metric>=<value>   runtime: <s>   status: <✅/📄/📝>
- command: scripts/repro/<script>.sh   (or inline)
- notes: <caveats, env, what changed>
```

---

### 2026-05-27 — PreJEPA dinov2_small trained on PushT (10 epochs)
- event: **training** (not an eval). `scripts/train/prejepa.py`
- model: `pusht_dinov2_small_psmall` (subdir `pusht_dinov2_repro`)
- command: `python scripts/train/prejepa.py dataset_name=pusht_expert_train.h5 subdir=pusht_dinov2_repro num_workers=8`
- result: 10 epochs, ~25 min/epoch, DDP; final val `pixels_loss≈0.152`, `proprio_loss≈0.0196`.
  Checkpoints `weights_epoch_{1,5,10}.pt` saved. wandb: disabled (trap #1).
- notes: full train config in `checkpoints/pusht_dinov2_repro/config.yaml`; weights in
  `checkpoints/pusht_dinov2_small_psmall/` (trap #2).

### 2026-05-27 — LeWM PushT eval (h5)
- model: `quentinll/lewm-pusht`@`weights.pt`   eval: `pusht-wm-cem`   seed: 42
- dataset: `pusht_expert_train.h5`
- result: **success_rate = 92.0%** (46/50)   runtime: 56.78 s   status: 📄
- command: `scripts/repro/lewm_pusht_eval.sh`
- notes: from `checkpoints/quentinll/pusht_results.txt` (written 09:23). Brackets the
  published ~96%. First eval after decompress threw a transient NFS KeyError; re-ran
  clean (trap #4). LeWM ViT keys were remapped to load strict (trap #5).

### 2026-05-27 — LeWM PushT eval (Lance) — reported, not on-disk here
- model: `quentinll/lewm-pusht`@`weights.pt`   eval: `pusht-wm-cem`   seed: 42
- dataset: `pusht_expert_train.lance`
- result: **success_rate = 98%** (49/50)   status: 📝
- notes: reported in `progress/repro_pushT.md`; no result file for it on this host. Same
  episodes sampled as the h5 run; spread is CEM/GPU nondeterminism. **Not verified here.**

### 2026-05-28 — PreJEPA dinov2_small PushT eval (h5)
- model: `pusht_dinov2_small_psmall`@`weights_epoch_10`   eval: `pusht-wm-cem`   seed: 42
- dataset: `pusht_expert_train.h5`
- result: **success_rate = 48.0%** (24/50)   runtime: 8728.18 s (≈2.4 h)   status: 📄
- command: `scripts/repro/prejepa_pusht_eval.sh`
- notes: from `checkpoints/pusht_dinov2_small_psmall/pusht_results.txt` (written 14:43).
  Only 10 epochs of training. Expensive eval (trap #3) — do not casually re-run.

### 2026-05-29 — DINO-WM PushT reproduction training (100 epochs) — IN PROGRESS
- event: **training** (not an eval). `scripts/train/prejepa.py`, paper recipe (papers/dinowm.pdf Table 12/11).
- model (will be): `pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep` (subdir/wandb id `dinowm_pusht_repro_100ep`)
- command: `scripts/repro/prejepa_pusht_train_100ep.sh`
- wandb: **enabled** (first run that logs) — https://wandb.ai/manugaur/stable-wm/runs/dinowm_pusht_repro_100ep
- config: AdamW, **lr=5e-5**, 100 epochs, batch 32/rank, **7-GPU DDP** (GPUs 1-7; global batch 224),
  H=3, frameskip=5, seed=42, dataset `pusht_expert_train.h5`. ~7962 steps/epoch, ~4.9 it/s
  → **~27 min/epoch, ~45 h total** (UNVERIFIED). Checkpoints every 5 epochs.
- judgement calls (vs paper / repo default):
  - **lr 5e-5** matches the paper's *predictor* lr (the 19M-param world model); repo default
    5e-4 trains the predictor 10× too hot — likely why the 10-epoch repro hit only 48%
    (paper's headline is 92%). The two tiny 10-dim embedders run at 5e-5 vs the paper's
    action-encoder 5e-4 — a full-fidelity reproduction would use a 2-optimizer split.
  - **7 GPUs not 8**: GPU 0 was running another user's job at launch; excluded to avoid
    DDP bottleneck/contention (it freed shortly after; not worth restarting).
  - decoder not trained — paper's 0.92 is "w/o decoder loss" (Table 7), which prejepa.py does.
- next: when done, eval `weights_epoch_100` with `scripts/repro/prejepa_pusht_eval.sh`
  (point it at the new checkpoint dir) and record success_rate here + in models.md.

### 2026-05-31 — DINO-WM 100ep repro PushT eval @ epoch 90 (h5)
- model: `pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep`@`weights_epoch_90`   eval: `pusht-wm-cem`   seed: 42
- dataset: `pusht_expert_train.h5`
- result: **success_rate = 8.0%** (4/50)   runtime: 10742.94 s (≈3.0 h)   status: ✅ (re-run fresh by us)
- command: `scripts/repro/dinowm_100ep_pusht_eval.sh`
- successes: episodes at array idx 17, 18, 20, 29 (of 50).
- notes: from `checkpoints/pusht_dinov2_small_psmall_dinowm_pusht_repro_100ep/pusht_results.txt`,
  re-run live on GPU 0. Evaluated at **epoch 90** (training run still in progress; epoch 95
  on disk, 100 not yet finished) per request — not the planned epoch-100.
  ⚠️ **ANOMALY:** this is the lr-corrected (5e-5) run that was *supposed* to fix the 10-epoch
  run's 48% and approach the paper's 92%. Instead it scored **8%** — far *worse* than both
  the 10-epoch run (48%, lr=5e-4) and the paper. CEM solve times were also huge (~5100–5600 s
  per reported solve). Needs investigation before trusting: candidate causes — lr 5e-5 too cold
  / underfit world model, val-loss-vs-MPC divergence, or a config mismatch between the two runs.
  Do **not** assume monotonic improvement with epochs; check val/mse curve on wandb
  (run `dinowm_pusht_repro_100ep`) and consider evaluating an earlier/later checkpoint.
  Expensive eval (trap #3) — budget before re-running.

### 2026-06-01 — ORIGINAL DINO-WM repo reproduction on PushT (training IN PROGRESS)
- event: **setup + training launch** using the upstream `dino_wm/` code (gaoyuezhou/dino_wm),
  NOT our `prejepa.py`. Goal: a trustworthy reference to debug the PreJEPA repro against
  ("their code vs our code, on identical data").
- model (will be): `dinowm_orig_pusht_f5h3` — frozen DINOv2 ViT-S/14 (**196** patches, note:
  DINO-WM resizes 224→196 before the backbone; PreJEPA used 256) + `ViTPredictor`
  (depth6/heads16/mlp2048, dim 404) + Conv1d proprio/action embedders. `has_decoder=False`.
- data: our `pusht_expert_train.h5` converted to DINO-WM's `pusht_noise/{train,val}` layout
  (lossless mp4 + .pth tensors; bit-exact vs h5). Train/val 16,816/1,869 (90/10, seed 42).
- recipe (faithful, dino_wm defaults): global **batch 32**, predictor **lr 5e-4**, frameskip 5,
  num_hist 3, num_pred 1, 100 epochs, train seed 0. (Deliberately NOT the prior run's
  batch 224 / lr 5e-5 — that batch/lr change is a confound.)
- speed: trains on **precomputed frozen DINOv2 features** (identical to on-the-fly; encoder
  frozen) held in /dev/shm; ~55,800 steps/epoch, ~18 it/s on 8 GPUs → **~50 min/epoch**
  (~3.5 days for 100). First diagnostic planned at **epoch 10** (vs PreJEPA@10ep = 48%).
- command: `scripts/repro/dinowm_orig_pusht_{dataprep,train}.sh`
- wandb: **`manugaur/stable-wm`**, run id `lvd4emdj` (name `dinowm_orig_pusht_f5h3_pusht_f5_h3_p1`).
  https://wandb.ai/manugaur/stable-wm/runs/lvd4emdj
- eval pipeline: `pusht-dinowm-plan` (dino_wm `plan.py`) **validated end-to-end** on the
  epoch-1 checkpoint (5 eps → ran clean, emitted `mpc/success_rate`). That 5-ep/epoch-1
  number is a *pipeline test only*, NOT a result.
- deviations (tracked in the `dino_wm/` clone): D2 `:05d` filenames, D3 recomputed
  STATE/PROPRIO stats, D5 guarded pointmaze import (no mujoco), D7 wandb→stable-wm,
  D8 py3.9→3.10 (upstream dinov2 hub), D9 decode-only-needed-frames (bit-identical),
  D10 `encode_obs` accepts precomputed feats, D11 `PushTFeatDataset`.
- next: eval `model_10.pth` with `pusht-dinowm-plan` (50 eps) → record SR here + models.md;
  then later epochs + cross-eval under `pusht-wm-cem`.

### 2026-06-02 — dinowm_orig PushT eval @ epoch 10 (native plan.py)
- model: `dinowm_orig_pusht_f5h3`@`model_10`   eval: `pusht-dinowm-plan`   seed: 99
- dataset: val goals from converted `pusht_noise` (goal 25 env-steps ahead); live PushT sim
- result: **mpc/success_rate = 32.0%** (16/50)   runtime: ≈4 h (contended w/ training on GPU0)   status: ✅
- command: `EPOCH=10 NEVALS=50 GPU=0 scripts/repro/dinowm_orig_pusht_plan_eval.sh`
- result file: `dino_wm/plan_outputs/20260602005803_dinowm_orig_pusht_f5h3_gH5/logs.json`
  (also mean_state_dist 71.2, mean_visual_dist 3.16, mean_proprio_dist 27.2).
- notes: **INTERIM** (epoch 10 of 100). DINO-WM's *own* code + eval on our **expert** data
  scores 32% at 10ep — far below the paper's ~0.92 (which is at epoch 100), so the headline
  comparison needs `model_100`. Loose vs PreJEPA@10ep (48%, but that's the swm `pusht-wm-cem`
  protocol + 256 patches — NOT the same eval, so not directly comparable). Lesson learned:
  running this eval on a training GPU dropped train throughput 18.4→6.5 it/s; **don't eval
  intermediate checkpoints during training** — finish training, then parallel-eval the curve.

---

> Next: investigate the 8% anomaly above (wandb val curve, earlier-epoch eval). Re-verify
> LeWM (cheap, ~1 min) to upgrade 📄→✅. wandb is now ON by default (trap #1 updated); the
> 100ep DINO-WM run is the first with a real run ID.
