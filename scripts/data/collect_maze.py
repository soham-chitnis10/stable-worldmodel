"""Collect offline datasets from diverse-maze environments via World.record_dataset()."""

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.diverse_maze import ExpertPolicy
from stable_worldmodel.policy import RandomPolicy


@hydra.main(version_base=None, config_path='./config', config_name='maze')
def run(cfg):
    """Run data collection for a diverse-maze environment."""

    world = swm.World(cfg.env_name, **cfg.world, render_mode='rgb_array')

    if cfg.get('policy', 'expert') == 'random':
        rng_pol = np.random.default_rng(cfg.seed)
        world.set_policy(
            RandomPolicy(seed=rng_pol.integers(0, 1_000_000).item())
        )
    else:
        world.set_policy(
            ExpertPolicy(
                action_noise=cfg.get('action_noise', 0.0),
                seed=cfg.seed,
            )
        )

    options = cfg.get('options')
    if options is not None:
        options = OmegaConf.to_object(options)

    rng = np.random.default_rng(cfg.seed)

    world.record_dataset(
        cfg.dataset_name,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    logging.success(
        f'Completed data collection for {cfg.env_name} '
        f'({cfg.num_traj} episodes -> {cfg.dataset_name})'
    )


if __name__ == '__main__':
    run()
