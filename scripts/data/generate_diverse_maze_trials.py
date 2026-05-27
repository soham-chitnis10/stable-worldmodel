"""Generate evaluation trials for diverse-maze environments.

All paths are resolved relative to the cache directory (STABLEWM_HOME or
cache_dir override), mirroring the layout used by collect_maze.py.

Basic usage:
    python scripts/data/generate_diverse_maze_trials.py

Override individual knobs:
    python scripts/data/generate_diverse_maze_trials.py \\
        dataset_name=maze2d_large_diverse_25maps \\
        maps_path=maze2d_large_diverse_25maps_train_maps.pt \\
        min_block_radius=9 max_block_radius=12

Multiple difficulty tiers:
    for tier in "5_8 9_12 13_16"; do
        min=$(echo $tier | cut -d_ -f1)
        max=$(echo $tier | cut -d_ -f2)
        python scripts/data/generate_diverse_maze_trials.py \\
            min_block_radius=$min max_block_radius=$max
    done

Point to a custom cache root:
    python scripts/data/generate_diverse_maze_trials.py \\
        cache_dir=/my/storage

Absolute paths still work:
    python scripts/data/generate_diverse_maze_trials.py \\
        maps_path=/abs/path/to/train_maps.pt \\
        dataset_name=maze2d_large_diverse_25maps
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from loguru import logger as logging

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.diverse_maze.evaluation.trial_generator import (
    TrialGenerator,
)


def _resolve_path(name_or_path: str, cache_datasets_dir: Path) -> Path:
    """Resolve a file reference against the cache datasets directory.

    - Absolute path that exists → returned as-is.
    - Relative path with a parent component → resolved from cwd.
    - Bare filename → looked up under *cache_datasets_dir*.
    """
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return cache_datasets_dir / p.name


@hydra.main(
    version_base=None, config_path='./config', config_name='maze_trials'
)
def run(cfg) -> None:
    cache_datasets_dir = get_cache_dir(
        cfg.get('cache_dir'), sub_folder='datasets'
    )

    dataset_path = cache_datasets_dir / f'{cfg.dataset_name}.h5'
    maps_path = _resolve_path(cfg.maps_path, cache_datasets_dir)
    output_path = cache_datasets_dir / cfg.output_name

    logging.info(f'Dataset : {dataset_path}')
    logging.info(f'Maps    : {maps_path}')
    logging.info(f'Output  : {output_path}')

    gen = TrialGenerator(
        env_name=cfg.env_name,
        dataset_path=dataset_path,
        maps_path=maps_path,
        n_trials=cfg.n_trials,
        min_block_radius=cfg.min_block_radius,
        max_block_radius=cfg.max_block_radius,
        seed=cfg.seed,
        unique_shortest_path=cfg.unique_shortest_path,
    )

    trials = gen.generate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trials, output_path)
    logging.success(f'Saved {len(trials["starts"])} trials → {output_path}')


if __name__ == '__main__':
    run()
