"""Collect offline datasets from diverse-maze environments.

Supports the full PLDM data-generation pipeline via World.record_dataset():
  - Map generation from scratch (MapGenerator) or loading pre-existing maps
  - Map exclusion for probe / OOD datasets
  - Map index filtering (start/end)
  - Generate-maps-only mode
  - Per-map sequential episode collection (N episodes per map)
  - Exploration policies: random, uniform, ou_trajectory
  - qvel prior, action repeat, and all PLDM config knobs

Usage examples:

  # Generate training dataset (25 maps, 2000 episodes/map, uniform policy)
  python collect_maze.py

  # Generate probe dataset excluding training maps
  python collect_maze.py \\
      dataset_name=maze2d_large_diverse_probe \\
      map.train_maps_n=20 \\
      map.exclude_map_path=/path/to/train_maps.pt \\
      n_episodes=50

  # Generate maps only (no data collection)
  python collect_maze.py gen_map_only=true

  # Use OU trajectory policy
  python collect_maze.py policy=ou_trajectory
"""

import hydra
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.diverse_maze import (
    UniformPolicy,
    OUTrajectoryPolicy,
    MapGenerator,
)
from stable_worldmodel.policy import RandomPolicy


# ------------------------------------------------------------------
# Map management
# ------------------------------------------------------------------

def _resolve_maps(cfg) -> tuple[dict | None, Path | None]:
    """Generate, load, or resolve the map catalogue.

    Returns:
        (maps_dict, maps_file_path) — maps_dict is ``{idx: map_key}`` or
        None if no maps are available.  maps_file_path is where the maps
        are saved on disk (for passing to World via ``maps_path``).
    """
    map_cfg = cfg.get("map", {})
    cache_datasets_dir = Path(get_cache_dir(cfg.get("cache_dir"), sub_folder="datasets"))

    def _resolve_map_file(name_or_path: str) -> Path:
        """Resolve a map file path: if it has no parent directory, look in the
        cache datasets directory."""
        p = Path(name_or_path)
        if p.is_absolute() and p.exists():
            return p
        # relative with directories (e.g. "foo/bar.pt") — try as-is first
        if p.exists():
            return p.resolve()
        # bare filename — resolve under cache dir
        resolved = cache_datasets_dir / p.name
        return resolved

    # load pre-existing maps
    maps = None
    map_path = map_cfg.get("map_path")
    if map_path is not None:
        map_path = _resolve_map_file(map_path)
        maps = torch.load(map_path, weights_only=False)
        logging.info(f"Loaded {len(maps)} maps from {map_path}")

    # load maps to exclude
    exclude_maps = None
    exclude_path = map_cfg.get("exclude_map_path")
    if exclude_path is not None:
        exclude_path = _resolve_map_file(exclude_path)
        raw = torch.load(exclude_path, weights_only=False)
        if isinstance(raw, dict):
            exclude_maps = {v: True for v in raw.values()} if all(isinstance(v, str) for v in raw.values()) else raw
        logging.info(f"Loaded {len(exclude_maps)} maps to exclude from {exclude_path}")

    # filter maps by index range
    if maps is not None:
        start_idx = map_cfg.get("map_start_idx")
        end_idx = map_cfg.get("map_end_idx")
        if start_idx is not None and end_idx is not None:
            maps = {k: v for k, v in maps.items() if start_idx <= int(k) <= end_idx}
            logging.info(f"Filtered to {len(maps)} maps (idx {start_idx}–{end_idx})")

    # truncate to train_maps_n
    train_maps_n = map_cfg.get("train_maps_n")
    if maps is not None and train_maps_n is not None and len(maps) > train_maps_n:
        maps = dict(list(maps.items())[:train_maps_n])

    # generate new maps if none loaded
    if maps is None:
        if train_maps_n is None or train_maps_n <= 0:
            logging.warning("No maps loaded and train_maps_n not set; skipping map generation")
            return None, None

        num_blocks = map_cfg.get("num_blocks_width_in_img", 12)
        generator = MapGenerator(
            width=num_blocks - 2,
            height=num_blocks - 2,
            num_maps=train_maps_n,
            sparsity_low=map_cfg.get("sparsity_low", 53),
            sparsity_high=map_cfg.get("sparsity_high", 88),
            max_path_len=map_cfg.get("max_path_len", 13),
            exclude_maps=exclude_maps,
            wall_coords=OmegaConf.to_object(map_cfg.get("wall_coords", [])),
            space_coords=OmegaConf.to_object(map_cfg.get("space_coords", [])),
        )
        maps = generator.generate_diverse_maps()
        logging.success(f"Generated {len(maps)} new maps")

    # save maps to disk
    output_dir = Path(
        get_cache_dir(cfg.get("cache_dir"), sub_folder="datasets")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_file = output_dir / f"{cfg.dataset_name}_train_maps.pt"
    torch.save(maps, maps_file)
    logging.info(f"Maps saved to {maps_file}")

    return maps, maps_file


# ------------------------------------------------------------------
# Policy factory
# ------------------------------------------------------------------

def _build_policy(cfg):
    """Instantiate a maze-collection policy from the Hydra config."""
    policy_name = cfg.get("policy", "uniform")

    if policy_name == "random":
        rng_pol = np.random.default_rng(cfg.seed)
        return RandomPolicy(seed=rng_pol.integers(0, 1_000_000).item())

    if policy_name == "ou_trajectory":
        ou = cfg.get("ou_params", {})
        return OUTrajectoryPolicy(
            ou_theta=ou.get("ou_theta", 0.15),
            ou_dt=ou.get("ou_dt", 1.0),
            ou_sigma=ou.get("ou_sigma", 0.2),
            turn_frequency=ou.get("turn_frequency", 20),
            turn_sigma=ou.get("turn_sigma", 0.2),
            turn_noise_level=ou.get("turn_noise_level", 0.05),
            seed=cfg.seed,
        )

    # default: uniform
    return UniformPolicy(
        max_norm=cfg.get("max_norm", 1.0),
        resample_every=cfg.get("resample_every", 1),
        seed=cfg.seed,
    )


# ------------------------------------------------------------------
# World construction helpers
# ------------------------------------------------------------------

def _world_env_kwargs(cfg, maps_file: Path | None) -> dict:
    """Build the extra kwargs forwarded to gym.make → DiverseMazeEnv."""
    kw = {}
    if maps_file is not None:
        kw["maps_path"] = str(maps_file)
    kw["action_repeat"] = cfg.get("action_repeat", 1)
    kw["action_repeat_mode"] = cfg.get("action_repeat_mode", "id")
    kw["qvel_prior"] = cfg.get("qvel_prior", False)
    kw["qvel_prior_type"] = cfg.get("qvel_prior_type", "uniform")
    return kw


# ------------------------------------------------------------------
# Collection modes
# ------------------------------------------------------------------

def _collect_with_variation(cfg, maps_file, maps):
    """Collect episodes with random variation sampling across all maps."""
    world_cfg = OmegaConf.to_object(cfg.world)
    world_cfg.pop("maps_path", None)
    env_kwargs = _world_env_kwargs(cfg, maps_file)

    world = swm.World(
        cfg.env_name,
        **world_cfg,
        render_mode="rgb_array",
        **env_kwargs,
    )
    world.set_policy(_build_policy(cfg))

    n_maps = len(maps) if maps else 1
    total_episodes = cfg.n_episodes * n_maps

    options = OmegaConf.to_object(cfg.get("options")) if cfg.get("options") else None

    rng = np.random.default_rng(cfg.seed)
    world.record_dataset(
        cfg.dataset_name,
        episodes=total_episodes,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.get("cache_dir"),
        options=options,
    )

    logging.success(
        f"Collected {total_episodes} episodes across {n_maps} maps "
        f"→ {cfg.dataset_name}"
    )


def _collect_per_map(cfg, maps_file, maps):
    """Collect exactly n_episodes per map, iterating maps sequentially.

    Each batch of episodes for a map is appended to the same HDF5 file
    via the resume capability of ``World.record_dataset``.
    """
    world_cfg = OmegaConf.to_object(cfg.world)
    world_cfg.pop("maps_path", None)
    env_kwargs = _world_env_kwargs(cfg, maps_file)

    world = swm.World(
        cfg.env_name,
        **world_cfg,
        render_mode="rgb_array",
        **env_kwargs,
    )
    world.set_policy(_build_policy(cfg))

    rng = np.random.default_rng(cfg.seed)
    cumulative_episodes = 0

    for map_idx, map_key in maps.items():
        cumulative_episodes += cfg.n_episodes
        logging.info(
            f"Collecting {cfg.n_episodes} episodes for map {map_idx} "
            f"(cumulative target: {cumulative_episodes})"
        )

        options = {
            "map_key": map_key,
            "map_idx": int(map_idx),
        }

        world.record_dataset(
            cfg.dataset_name,
            episodes=cumulative_episodes,
            seed=rng.integers(0, 1_000_000).item(),
            cache_dir=cfg.get("cache_dir"),
            options=options,
        )

    total = cfg.n_episodes * len(maps)
    logging.success(
        f"Collected {total} episodes ({cfg.n_episodes} per map × {len(maps)} maps) "
        f"→ {cfg.dataset_name}"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config", config_name="maze")
def run(cfg):
    """Run diverse-maze data collection."""

    # Step 1: map management
    maps, maps_file = _resolve_maps(cfg)

    # Step 2: optional early exit (maps only)
    if cfg.get("gen_map_only", False):
        logging.success("gen_map_only=true — maps generated, skipping data collection.")
        return

    if maps is None or len(maps) == 0:
        logging.error("No maps available for data collection. Exiting.")
        return

    # Step 3: data collection
    collect_mode = cfg.get("collect_mode", "per_map")
    if collect_mode == "variation":
        _collect_with_variation(cfg, maps_file, maps)
    else:
        _collect_per_map(cfg, maps_file, maps)


if __name__ == "__main__":
    run()
