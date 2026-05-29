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

---

> Next: re-verify LeWM (cheap, ~1 min) to upgrade 📄→✅; enable wandb before the next
> training run so it gets a real run ID.
