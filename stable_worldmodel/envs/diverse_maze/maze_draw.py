"""Top-down paper-scale drawer for gymnasium-robotics point mazes.

``DiverseMazeEnv.render()`` delegates here. The drawer:

1. Forces the inner ``PointEnv`` MuJoCo renderer into a top-down camera
   (installed by :func:`maze_physics._apply_top_down_camera`).
2. Applies env-specific ``select_transforms`` (center-crop + resize) to emit
   the paper-scale ``(H, W, 3)`` ``uint8`` frame used in HJEPA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from stable_worldmodel.envs.diverse_maze.maze_physics import set_point_state
from stable_worldmodel.envs.diverse_maze.transforms import select_transforms
from stable_worldmodel.envs.diverse_maze.maze_stats import RENDER_STATS
from stable_worldmodel.envs.diverse_maze.utils import PixelMapper


class GymnasiumPointMazeDrawer:
    """Top-down RGB via gymnasium-robotics ``PointMazeEnv`` / ``point_env``.

    Matches HJEPA by applying env-specific ``select_transforms`` (center-crop
    + resize) to the raw MuJoCo frame so ``env.render()`` returns a paper-scale
    ``(H, W, 3)`` ``uint8`` array.
    """

    def __init__(self, env: Any, env_id: str):
        self.env = env
        self.env_id = env_id
        self.point_env = getattr(env, 'point_env', None)
        if self.point_env is None:
            raise TypeError(
                'Expected PointMazeEnv with .point_env for rendering'
            )
        self._transform = select_transforms(
            getattr(env, 'name', env_id) or env_id
        )

    def render_state(self, obs: Any) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).ravel()
        set_point_state(self.env, obs.astype(np.float64))
        img = self.point_env.render()
        frame = np.asarray(img, dtype=np.uint8)
        if self._transform is None:
            return frame
        pil = Image.fromarray(frame)
        transformed = self._transform(pil)
        return np.asarray(transformed, dtype=np.uint8)

    @torch.no_grad()
    def __call__(self, obs: Any, ax: Any = None) -> Any:
        import matplotlib.pyplot as plt

        obs_t = torch.as_tensor(obs, dtype=torch.float32).data.cpu()[:4]
        rgb = self.render_state(obs_t.numpy())
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb)
        ax.autoscale(False)
        stats = RENDER_STATS.get(
            self._stats_env_id(),
            {'lookat': [3.5, 3.5, 0], 'arrow_mult': 25.0},
        )
        pm = PixelMapper(env_name=self.env.name)
        ax.arrow(
            *pm.obs_coord_to_pixel_coord(obs_t[:2]),
            *obs_t[2:]
            .mul(torch.tensor([1, -1], dtype=torch.float32))
            .mul(float(stats.get('arrow_mult', 25.0))),
            width=6,
            color='red',
            alpha=0.6,
        )
        return ax

    def _stats_env_id(self) -> str:
        eid = getattr(self.env, 'name', self.env_id)
        if 'diverse' in eid:
            i = eid.find('diverse')
            eid = eid[: i + len('diverse')]
        return eid


def create_drawer(env: Any, env_id: str) -> GymnasiumPointMazeDrawer:
    """Build a drawer for a gymnasium-robotics point maze env."""
    if env is None:
        import gymnasium as gym

        env = gym.make(env_id)
        env.name = env_id
    return GymnasiumPointMazeDrawer(env, env_id)
