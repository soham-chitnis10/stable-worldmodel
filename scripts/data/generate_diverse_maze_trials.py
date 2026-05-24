"""Generate evaluation trials for diverse-maze environments.

Usage (single difficulty tier):
    python scripts/data/generate_diverse_maze_trials.py \\
        --env_name maze2d_large_diverse \\
        --dataset_path /path/to/dataset \\
        --maps_path /path/to/train_maps.pt \\
        --output_path /path/to/trials.pt \\
        --n_trials 80 \\
        --min_block_radius 5 \\
        --max_block_radius 8

Multiple difficulty tiers (easy / medium / hard):
    for tier in "5_8 9_12 13_16"; do
        min=$(echo $tier | cut -d_ -f1)
        max=$(echo $tier | cut -d_ -f2)
        python scripts/data/generate_diverse_maze_trials.py \\
            --dataset_path /path/to/dataset \\
            --maps_path /path/to/train_maps.pt \\
            --output_path trials_${min}_${max}.pt \\
            --min_block_radius $min --max_block_radius $max
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stable_worldmodel.envs.diverse_maze.evaluation.trial_generator import TrialGenerator


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate diverse-maze evaluation trials (starts/targets/map_layouts)"
    )
    p.add_argument("--env_name", default="maze2d_large_diverse")
    p.add_argument("--dataset_path", required=True, help="Collected diverse-maze dataset")
    p.add_argument("--maps_path", required=True, help="train_maps.pt torch dict")
    p.add_argument("--output_path", required=True, help="Where to write trials.pt")
    p.add_argument("--n_trials", type=int, default=80)
    p.add_argument("--min_block_radius", type=int, default=5)
    p.add_argument("--max_block_radius", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unique_shortest_path", action="store_true")
    args = p.parse_args()

    gen = TrialGenerator(
        env_name=args.env_name,
        dataset_path=args.dataset_path,
        maps_path=args.maps_path,
        n_trials=args.n_trials,
        min_block_radius=args.min_block_radius,
        max_block_radius=args.max_block_radius,
        seed=args.seed,
        unique_shortest_path=args.unique_shortest_path,
    )

    trials = gen.generate()

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trials, out)
    print(f"Saved {len(trials['starts'])} trials → {out}")


if __name__ == "__main__":
    main()
