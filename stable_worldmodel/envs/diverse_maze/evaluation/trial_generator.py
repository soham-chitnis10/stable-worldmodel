"""Generate (start, target, map_layout, block_dist, turns) trial dicts
for diverse-maze evaluation.

Starts are sampled from a collected dataset — guaranteeing they are in
valid, reachable positions.  Targets are sampled via BFS on the maze
layout within a configurable block-distance window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_worldmodel.envs.diverse_maze.maze_stats import RENDER_STATS
from stable_worldmodel.envs.diverse_maze.utils import sample_nearby_grid_location_v2


def _stats_key(env_name: str) -> str:
    if "diverse" in env_name:
        end = env_name.find("diverse") + len("diverse")
        return env_name[:end]
    return env_name


def _load_states_by_map(dataset_path: str | Path) -> dict[int, np.ndarray]:
    """Load (x, y, vx, vy) states from a dataset, keyed by map_idx.

    Returns:
        {map_idx: ndarray of shape (N, 4)}
    """
    from stable_worldmodel.data import load_dataset

    ds = load_dataset(str(dataset_path))
    states = ds.get_col_data("state")              # (N, 4)
    map_ids = ds.get_col_data("map_idx").ravel()   # (N,)

    by_map: dict[int, np.ndarray] = {}
    for idx in np.unique(map_ids):
        key = int(idx)
        by_map[key] = states[map_ids == idx]
    return by_map


class TrialGenerator:
    """Generate reproducible (start, target) trials for diverse-maze eval.

    Parameters
    ----------
    env_name : str
        Base env name, e.g. ``maze2d_large_diverse``.
    dataset_path : str | Path
        Path to a collected diverse-maze dataset (lance / hdf5 / folder).
    maps_path : str | Path
        Path to a ``train_maps.pt`` torch dict ``{idx: map_key}``.
    n_trials : int
        Total number of trials to generate.
    min_block_radius, max_block_radius : int
        BFS distance window for target sampling.
    seed : int
        RNG seed for reproducibility.
    unique_shortest_path : bool
        If True, prefer targets whose shortest path is unique.
    """

    def __init__(
        self,
        env_name: str,
        dataset_path: str | Path,
        maps_path: str | Path,
        n_trials: int,
        min_block_radius: int,
        max_block_radius: int,
        seed: int = 42,
        unique_shortest_path: bool = False,
    ) -> None:
        stats = RENDER_STATS[_stats_key(env_name)]
        self.obs_range_total: float = stats["obs_range_total"]
        self.obs_min_total: float = stats["obs_min_total"]

        self.n_trials = n_trials
        self.min_block_radius = min_block_radius
        self.max_block_radius = max_block_radius
        self.unique_shortest_path = unique_shortest_path
        self.rng = np.random.default_rng(seed)

        self.maps: dict = torch.load(str(maps_path), weights_only=False)
        self.states_by_map = _load_states_by_map(dataset_path)

    def generate(self) -> dict[str, list[Any]]:
        """Generate and return a trials dict.

        Keys: ``starts``, ``targets``, ``map_layouts``, ``block_dists``, ``turns``.
        """
        trials: dict[str, list] = {
            "starts": [],
            "targets": [],
            "map_layouts": [],
            "block_dists": [],
            "turns": [],
        }

        map_keys = list(self.maps.keys())

        for i in range(self.n_trials):
            map_idx = map_keys[i % len(map_keys)]
            map_key = self.maps[map_idx]

            pool = self.states_by_map.get(int(map_idx))
            if pool is None or len(pool) == 0:
                raise ValueError(f"No states for map_idx={map_idx} in dataset")

            start = pool[self.rng.integers(len(pool))]  # (4,) — x, y, vx, vy

            target, block_dist, turns, _ = sample_nearby_grid_location_v2(
                anchor=start[:2].astype(float),
                map_key=map_key,
                min_block_radius=self.min_block_radius,
                max_block_radius=self.max_block_radius,
                obs_range_total=self.obs_range_total,
                obs_min_total=self.obs_min_total,
                unique_shortest_path=self.unique_shortest_path,
            )

            trials["starts"].append(start.copy())
            trials["targets"].append(np.asarray(target, dtype=np.float32))
            trials["map_layouts"].append(map_key)
            trials["block_dists"].append(int(block_dist))
            trials["turns"].append(int(turns))

        return trials
