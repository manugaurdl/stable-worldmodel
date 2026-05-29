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

## Not yet trained

No models exist for `pldm`, `tdmpc2`, `gcrl` (gcbc/gciql/gcivl/hilp), or for any non-PushT
env (cube / reacher / tworoom). Add a section here when one is trained + evaluated.
