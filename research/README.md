# Research orientation — stable-worldmodel

**Read this first every session.** It says what the project is, how the workflow runs,
what models exist, the key results, and what's next. For rules see
[`../CLAUDE.md`](../CLAUDE.md); for exact commands see [`reference.md`](reference.md);
authoritative numbers live in [`models.md`](models.md) +
[`reproduction-log.md`](reproduction-log.md).

---

## What this project is

We use **stable-worldmodel** (a platform for *collect data → train world model →
evaluate with MPC*) to **develop a novel world-model architecture / training objective**.
The endpoint is a model + objective that plans better than existing world models on
manipulation tasks.

Two existing models are our **baselines / groundwork**, not the contribution:
- **LeWM** — a latent world model with a public pretrained PushT checkpoint. We
  reproduced its published planning success rate to validate the eval pipeline.
- **PreJEPA** — the repo's DINO-WM reproduction (frozen DINO backbone + causal
  predictor). We trained one PushT model as a baseline to build on.

Everything is benchmarked on **PushT** for now (`swm/PushT-v1`): a 2D pushing task where
success = block within 20 px and 20° of the goal pose. The headline metric is
**`success_rate`** (% of 50 episodes solved within a 50-step budget under CEM planning).

## The workflow (collect → train → evaluate)

1. **Collect / get data.** PushT expert data lives at
   `$STABLEWM_HOME/datasets/pusht_expert_train.h5` (18,685 episodes, ~2.34 M steps).
   Loader autodetects h5 / Lance / folder / video.
2. **Train** a world model: `python scripts/train/<model>.py ...` (Hydra). Saves
   `weights_epoch_{N}.pt` + `config.json` to `$STABLEWM_HOME/checkpoints/<run>/`.
3. **Evaluate with MPC:** `python scripts/plan/eval_wm.py policy=<model-id> ...`. The
   model is wrapped in a CEM solver + `WorldModelPolicy`; the *real* PushT simulator
   decides success. Writes `pusht_results.txt` + rollout videos next to the checkpoint.

Exact commands, templates, paths, and gotchas: [`reference.md`](reference.md).

## Current models

| Model ID | Type | Checkpoint | Trained here? | wandb |
|---|---|---|---|---|
| `quentinll/lewm-pusht` | LeWM (pretrained, HF) | `weights.pt` (ViT keys remapped) | No — downloaded | — |
| `pusht_dinov2_small_psmall` | PreJEPA / DINO-WM repro: frozen dinov2_small + CausalPredictor(small), 10 epochs | `weights_epoch_{1,5,10}.pt` | Yes (2026-05-27) | none (see trap #1) |

Full entries (config, train cmd, reproduce cmd) in [`models.md`](models.md).
> Note: the PreJEPA run's full train `config.yaml` is in the *sibling* dir
> `pusht_dinov2_repro/`, not the checkpoint dir — same run (CLAUDE.md trap #2).

## Key results (TL;DR)

All on PushT, eval `pusht-wm-cem` (CEM MPC, 50 eps, seed 42, h5 dataset).
**Both numbers are transcribed from on-disk result files — not yet re-verified.**

| Model @ checkpoint | success_rate | Notes |
|---|---|---|
| `quentinll/lewm-pusht` @ `weights.pt` | **92.0%** | h5; brackets the published ~96%. Progress doc also reports 98% on Lance (not re-verified). Eval ~57 s. |
| `pusht_dinov2_small_psmall` @ `weights_epoch_10` | **48.0%** | h5; only 10 epochs of training. Eval ~2.4 h (trap #3). |

Authoritative copies + reproduce commands: [`models.md`](models.md).

> **Working journal:** day-to-day progress (what I'm doing, why, narrative + results) is
> kept in [`../progress/*.md`](../progress) — e.g. the active reproduction is logged in
> [`../progress/repro_pushT.md`](../progress/repro_pushT.md). Those are the story; the
> authoritative numbers + reproduce commands live in `models.md` /
> `reproduction-log.md`, which the journal cites.

## Active todos

- [ ] **Enable wandb logging** (`wandb.enabled=true wandb.config.entity=manugaur`) so
      future training runs land in project `stable-wm` and get real model IDs = `subdir`
      (CLAUDE.md trap #1).
- [ ] **Re-verify both numbers fresh** (currently transcribed): LeWM is cheap (~1 min);
      PreJEPA is ~2.4 h.
- [ ] Train PreJEPA longer / larger and beyond 10 epochs to see where 48% goes.
- [ ] Decide the novel-objective direction and set up the first experiment against the
      PreJEPA baseline.

## Map of the research docs

| File | Role |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | Invariants, before-reporting checklist, traps log |
| `README.md` (this file) | Orientation — read first |
| [`reference.md`](reference.md) | Static reference: eval IDs, commands, paths, env setup, gotchas |
| [`models.md`](models.md) | Source of truth: one entry per model |
| [`reproduction-log.md`](reproduction-log.md) | Source of truth: dated repro log |
| [`../progress/`](../progress) | Working journal + presentation — what I'm doing, why, narrative + cited results per effort |
| `../scripts/repro/` | One reproduce script per result |
