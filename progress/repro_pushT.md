# Reproducing LeWM on Push-T

**Goal:** evaluate a pretrained world model on Push-T, reproduce the published
success rate, and learn the codebase. No training.

**Why LeWM (not DINO-WM):** DINO-WM has no public checkpoint (every row in
`docs/baselines.md` says "Checkpoint: NA"). LeWM ships a public checkpoint, so it's
the right model for an "evaluate a pretrained model" first pass.

**Result:** reproduced. Published number is 96%; I got **h5 = 92% (46/50)** and
**Lance = 98% (49/50)** with seed 42. Both runs sample the same episodes; the small
spread is CEM/GPU nondeterminism. Both bracket 96%.

---

## What I did

### 0. Environment
```bash
uv venv --python=3.10 && source .venv/bin/activate
uv sync --all-extras --group dev
```
Then patched a few things the install missed (see Gotchas).
All runs use `export STABLEWM_HOME=/data3/mgaur/stable_worldmodel` so datasets and
checkpoints live on `/data3` (1.4T free), not the default `~/.stable_worldmodel`.

### 1. Dataset
Downloaded the expert Push-T data from HuggingFace, decompressed it, and converted to
Lance (the loader autodetects format; h5 works too).
```bash
# download (HF dataset repo quentinll/lewm-pusht), then:
zstd -d --long=27 -o $STABLEWM_HOME/datasets/pusht_expert_train.h5 pusht_expert_train.h5.zst   # ~44G
swm convert pusht_expert_train pusht_expert_train.lance --dest-format lance                     # ~5.6G, 18,685 episodes
```
The dataset is only used to sample episode start states and goal states (25 steps
ahead). The pixels the model sees come from rendering the live env, not the dataset.

### 2. Checkpoint
The checkpoint is `quentinll/lewm-pusht` on HuggingFace (`config.json` + `weights.pt`,
72M). The eval script auto-downloads it, but its downloader uses `urllib` which fails
on SSL here, so I fetched the two files with `curl` into the cache folder
`$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht/`.

The weights didn't load directly: they were saved with the **old** HuggingFace ViT
layer names, but the installed `transformers` builds the **new** refactored ViT. Same
12 layers and shapes, only the names differ. I remapped the names and confirmed a
strict load (0 missing / 0 unexpected keys), then saved the fixed `weights.pt`
(original backed up to `$STABLEWM_HOME/_backups/`).

Name mapping (per transformer block, `encoder.encoder.layer.{i}.*` → `encoder.layers.{i}.*`):
`attention.attention.query/key/value` → `attention.q/k/v_proj`,
`attention.output.dense` → `attention.o_proj`,
`intermediate.dense` → `mlp.fc1`, `output.dense` → `mlp.fc2`.

### 3. Evaluate
```bash
export STABLEWM_HOME=/data3/mgaur/stable_worldmodel CUDA_VISIBLE_DEVICES=0 SDL_VIDEODRIVER=dummy
python scripts/plan/eval_wm.py policy=quentinll/lewm-pusht eval.dataset_name=pusht_expert_train.h5
# or .lance
```
Defaults (`scripts/plan/config/pusht.yaml`): 50 episodes, 50-step budget, goal 25 steps
ahead, CEM solver (300 samples, 30 iters, top-30), plan horizon 5 / action block 5.
Outputs: 50 rollout videos + `pusht_results.txt` in
`$STABLEWM_HOME/checkpoints/quentinll/`.

---

## Gotchas / fixes
- **Missing deps** not in any extra: `uv pip install scikit-learn datasets`.
- **Broken pygame** (crashes on import, missing `FRect`): `uv pip install --reinstall "pygame==2.6.1"`.
- **Headless rendering**: set `SDL_VIDEODRIVER=dummy`.
- **SSL**: the checkpoint/dataset downloaders use `urllib` and fail cert verification; use `curl` instead.
- **Checkpoint key mismatch**: remap old→new ViT layer names (above).
- **Lance support in eval_wm.py** (2 edits, working tree only, not committed):
  1. Lance hides index columns from `column_names`, so the
     `'episode_idx' if ... else 'ep_idx'` heuristic picked the wrong name. Flipped it to
     default to `episode_idx`.
  2. `get_row_data(...)['episode_idx'/'step_idx']` KeyErrors on Lance; replaced with
     `get_col_data(col)[indices]` (works on both formats).

---

## How the eval works (call path)
```
eval_wm.py
  -> load_pretrained("quentinll/lewm-pusht")          # build LeWM, load weights
  -> World("swm/PushT-v1", num_envs=50).evaluate(...)  # set start+goal from dataset
       loop:
         WorldModelPolicy.get_action(obs)              # receding-horizon MPC
           -> CEMSolver.solve(...)                      # sample action seqs, keep elites
                -> LeWM.get_cost(obs, actions)          # encode goal, rollout, MSE cost
       success = env terminated (block within 20px and 20 deg of goal)
```
Model code to study/modify for later architecture work: `stable_worldmodel/wm/lewm/lewm.py`.
