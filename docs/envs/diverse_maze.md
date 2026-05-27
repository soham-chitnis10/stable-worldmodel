---
title: Diverse maze2d
summary: Multi-layout Gymnasium point mazes (maze2d small / medium / large diverse)
sidebar_title: Diverse maze2d
---

## Overview

These environments wrap **gymnasium-robotics** point mazes with multiple wall
layouts (`maze.map_idx` variation). They follow the same patterns as other SWM
envs: **`World.record_dataset`**, reset/step **`info`** with **`state`**,
**`goal_state`**, **`proprio`**, **`map_idx`**, and **`env_name`**.

Registered ids:

| Gym id | Render shape | Notes |
|--------|--------------|-------|
| `swm/maze2d_small_diverse`  | `(64, 64, 3)` | Small grid  |
| `swm/maze2d_medium_diverse` | `(81, 81, 3)` | Medium grid |
| `swm/maze2d_large_diverse`  | `(98, 98, 3)` | Large grid  |

```python
import gymnasium as gym
import stable_worldmodel.envs  # registers swm/*

env = gym.make("swm/maze2d_large_diverse", render_mode="rgb_array")
obs, info = env.reset(seed=0)
# info: state (4,), goal_state (2,), proprio (2,), map_idx, env_name
img = env.render()  # (98, 98, 3) uint8, top-down paper-scale frame
```

## Rendering

`DiverseMazeEnv.render()` returns a **paper-scale top-down** frame that matches
HJEPA conventions:

- The inner `PointEnv` MuJoCo camera is pinned to `elevation=-90`, `azimuth=90`,
  `lookat=(0, 0, 0)` at env construction
  (see `stable_worldmodel/envs/diverse_maze/maze_physics.py`).
- `GymnasiumPointMazeDrawer` applies `select_transforms(env.name)` — a
  `CenterCrop` + `Resize` pipeline per env root — to emit the paper-scale
  `(H, W, 3)` `uint8` array shown in the table above.

## Exploration policies

Two exploration policies are available for data collection:

- **`UniformPolicy`** — random 2-D actions with bounded norm, optionally held
  for `resample_every` consecutive steps.
- **`OUTrajectoryPolicy`** — Ornstein-Uhlenbeck noise with occasional
  velocity-aware turn manoeuvres; mirrors the PLDM data-generation pipeline.

Both expose `get_action(info_dict)` and accept a `seed` argument.

## Map catalog (`train_maps.pt`)

Layouts are loaded from a **`train_maps.pt`** torch dict (keys → maze string
layouts). Resolution order in `DiverseMazeEnv`:

1. Explicit **`maps_path`** passed to `gym.make(..., maps_path=...)` or
   `make_diverse_maze_env(..., maps_path=...)`.
2. Small **in-repo** defaults under
   `stable_worldmodel/envs/diverse_maze/presaved_datasets/` when present.
3. **`get_cache_dir(sub_folder="diverse_maze") / "{env_name}_train_maps.pt"`**
   (same pattern as other datasets in `stable_worldmodel.data.utils`).

Layout characters follow gymnasium-robotics semantics:

- `#` — wall
- `r` — reset-only cell
- `g` — goal-only cell
- `c` — combined reset/goal cell
- `O` / `0` — empty cell (free)

For CI or local tests, point `maps_path` at a tiny fixture (see
`tests/envs/test_diverse_maze.py`).

## Data collection

Hydra entry point: **`scripts/data/collect_maze.py`**
(config: **`scripts/data/config/maze.yaml`**).

```bash
cd stable-worldmodel
python scripts/data/collect_maze.py \
  env_name=swm/maze2d_small_diverse \
  policy=expert \
  num_traj=1 \
  world.maps_path=/path/to/train_maps.pt
```

Optional keys under **`world`** (forwarded to `gym.make`): **`maps_path`**,
**`max_episode_steps`**, **`num_envs`**, **`image_shape`**, etc.

## Planning defaults

Planning Hydra config **`scripts/plan/config/maze.yaml`** uses
`swm/maze2d_small_diverse` and **`dataset_name: maze2d_small_diverse`**,
aligned with **`scripts/data/config/maze.yaml`**.
