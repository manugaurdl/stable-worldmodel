# Understanding `eval_wm.py` — `run()`

## High-level flow of `run()`

`run()` does **setup → mine eval tasks → run → save**:

1. **Build the world** (the real simulator, 50 parallel Push-T envs).
2. **Build the model + planner** (load LeWM checkpoint → CEM solver → `WorldModelPolicy`).
3. **Mine eval tasks from the dataset** (pick 50 valid `(episode, start_step)` pairs; goal = 25 steps later).
4. **Run evaluation** (`world.evaluate`) and **save metrics/videos**.

Everything before `world.evaluate` is preparation; the actual eval is one call.

---

## Block-by-block

### Block 0 — Sanity check (`:57-60`)
```python
assert horizon * action_block <= eval_budget
```
- A plan covers `horizon × action_block` = 5×5 = 25 env steps. That must fit inside the `eval_budget` = 50 step allowance. Guards against asking for a plan longer than the episode.

### Block 1 — Build the world (`:62-64`)
```python
cfg.world.max_episode_steps = 2 * eval_budget
world = swm.World(**cfg.world, image_shape=(224,224))
```
- Creates 50 parallel `swm/PushT-v1` envs (the **real** simulator that decides success).
- Episode cap set to 2×budget (slack). Renders at 224×224 (ViT input size).

### Block 2 — Image transform (`:66-71`)
```python
transform = {'pixels': img_transform(...), 'goal': img_transform(...)}
```
- The preprocessing applied to env frames **and** goal frames before the model sees them: ToImage → float → **ImageNet normalize** → resize. Same transform for current and goal images so they live in the same space.

### Block 3 — Dataset + normalizers (`:73-93`)
```python
dataset = get_dataset(...)
ep_indices = unique(episode_idx)
process = {col: StandardScaler().fit(...) for col in keys_to_cache}
```
- Loads the Push-T dataset.
- Fits a `StandardScaler` per non-pixel column (`state`, `proprio`, `action`) → these normalize/denormalize vectors the model works with.
- Also registers `goal_<col>` scalers (reuses the same fit) so goal vectors are scaled identically. (NaN rows dropped before fitting.)

### Block 4 — Build model + solver + policy (`:96-120`)
```python
model = load_pretrained(cfg.policy).cuda().eval(); model.requires_grad_(False)
config = PlanConfig(**cfg.plan_config)
solver = instantiate(cfg.solver, model=model)         # CEMSolver
policy = WorldModelPolicy(solver, config, process, transform)
```
- Loads the frozen LeWM checkpoint (or `RandomPolicy` if `policy=random`).
- Wraps it: `PlanConfig` (horizon/receding/action_block) → `CEMSolver` → `WorldModelPolicy`. This `policy` is what the world will query each step.
- (Optional `bf16`/`torch.compile` branches for speed.)

### Block 5 — Results path (`:125-131`)
- Decides where videos + metrics file go (next to the checkpoint folder).

### Block 6 — Mine valid start points (`:134-150`)
```python
episode_len = get_episodes_length(...)
max_start_idx = episode_len - goal_offset_steps - 1
valid_mask = step_idx <= max_start_per_row
valid_indices = nonzero(valid_mask)
```
- Goal is always **25 steps after start**, so a start step is only valid if the episode has ≥25 steps left.
- Computes each episode's length, then marks every dataset row whose `step_idx` leaves room for a 25-step-ahead goal. `valid_indices` = all usable start rows.

### Block 7 — Sample 50 eval tasks (`:152-165`)
```python
random_episode_indices = rng.choice(valid_indices, num_eval=50, replace=False)
sorted...
eval_episodes  = episode_idx[random_episode_indices]
eval_start_idx = step_idx[random_episode_indices]
```
- Seeded RNG picks 50 valid start rows (sorted for HDF5 indexing).
- Resolves them into the two arrays `world.evaluate` needs: which **episode** and which **start step** in it. (Goal step is derived later as start+25.)

### Block 8 — Run evaluation (`:172-220`)
```python
world.set_policy(policy)
metrics = world.evaluate(
    dataset, start_steps=eval_start_idx, goal_offset=25,
    eval_budget=50, episodes_idx=eval_episodes, callables=..., video=...)
```
- The real work (the next file to descend into). `callables` tells the env how to set start state and goal state from the dataset.
- (An optional `compile` warmup runs one batch first.)

### Block 9 — Save (`:223-238`)
- Prints `metrics` (success rate), writes config + metrics + timing to a results file.

---

The one line that matters for everything downstream is **`world.evaluate(...)` at `:210`**.
