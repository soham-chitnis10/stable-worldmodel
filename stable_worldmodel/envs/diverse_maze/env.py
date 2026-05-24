from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from stable_worldmodel import spaces as swm_spaces

from scipy.stats import truncnorm

from stable_worldmodel.envs.diverse_maze.maze_physics import (
    load_environment,
    set_point_state,
)

DEFAULT_VARIATIONS = ('maze.map_idx',)

# qvel truncated-normal distribution defaults (matches PLDM)
_QVEL_TRUNCNORM_LOWER = -5.2
_QVEL_TRUNCNORM_UPPER = 5.2
_QVEL_TRUNCNORM_MEAN = 0.0
_QVEL_TRUNCNORM_STD = 1.6

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Map catalogue helpers
# ------------------------------------------------------------------


def _default_maps_path(env_name: str) -> Path | None:
    base = Path(__file__).resolve().parent / 'presaved_datasets'
    if 'small' in env_name:
        return base / '5maps' / 'train_maps.pt'
    if 'large' in env_name:
        return base / '40maps' / 'train_maps.pt'
    if 'medium' in env_name:
        return base / '20maps' / 'train_maps.pt'
    return None


def _cache_maps_path(env_name: str) -> Path:
    from stable_worldmodel.data.utils import get_cache_dir

    return (
        get_cache_dir(sub_folder='diverse_maze') / f'{env_name}_train_maps.pt'
    )


@lru_cache(maxsize=8)
def _load_maps(path: str) -> dict:
    return torch.load(path)


# ------------------------------------------------------------------
# DiverseMazeEnv
# ------------------------------------------------------------------


class DiverseMazeEnv(gym.Env):
    """Diverse maze environment adapted for the stable-worldmodel framework.

    Wraps gymnasium-robotics point-maze environments with:
    - A ``variation_space`` exposing ``maze.map_idx`` for domain randomisation.
    - Rendering via ``maze_draw`` (proper camera, crop, resize).
    - Convenience ``_set_state`` / ``_set_goal_state`` for planning.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        env_name: str,
        map_key: str | None = None,
        map_idx: int | None = None,
        block_dist: int | None = None,
        turns: int | None = None,
        max_episode_steps: int = 600,
        render_mode: str = 'rgb_array',
        maps_path: str | Path | None = None,
        action_repeat: int = 1,
        action_repeat_mode: str = 'id',
        qvel_prior: bool = False,
        qvel_prior_type: str = 'uniform',
    ) -> None:
        super().__init__()

        self.env_name = env_name
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._block_dist = block_dist
        self._turns = turns
        self._maps_path_override = maps_path

        # action repeat (frame-skip in physics)
        self._action_repeat = int(action_repeat)
        self._action_repeat_mode = str(action_repeat_mode)

        # initial velocity prior applied at each reset
        self._qvel_prior = bool(qvel_prior)
        self._qvel_prior_type = str(qvel_prior_type)
        if self._qvel_prior and self._qvel_prior_type == 'normal':
            a = (
                _QVEL_TRUNCNORM_LOWER - _QVEL_TRUNCNORM_MEAN
            ) / _QVEL_TRUNCNORM_STD
            b = (
                _QVEL_TRUNCNORM_UPPER - _QVEL_TRUNCNORM_MEAN
            ) / _QVEL_TRUNCNORM_STD
            self._qvel_dist = truncnorm(
                a, b, loc=_QVEL_TRUNCNORM_MEAN, scale=_QVEL_TRUNCNORM_STD
            )

        # --- map catalogue (diverse envs only) ---
        self._maps: dict = {}
        self._map_index_to_key: list = []
        if 'diverse' in env_name:
            catalog = self._resolve_maps_catalog_path()
            if catalog is not None:
                self._maps = _load_maps(str(catalog))
                self._map_index_to_key = sorted(
                    self._maps.keys(), key=lambda v: str(v)
                )

        # --- resolve initial map ---
        self._current_map_key: str | None = map_key
        self._current_map_idx: int | None = map_idx
        if map_key is None and self._maps:
            pos = self._map_pos_for(map_idx)
            real_key = self._map_index_to_key[pos]
            self._current_map_key = self._maps[real_key]
            self._current_map_idx = (
                int(real_key) if str(real_key).isdigit() else pos
            )

        # --- inner physics env ---
        self._env = self._build_inner_env()
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # --- variation space ---
        self.variation_space = self._build_variation_space()

        # --- lazy drawer for maze_draw rendering ---
        self._drawer = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _resolve_maps_catalog_path(self) -> Path | None:
        if self._maps_path_override is not None:
            p = Path(self._maps_path_override).expanduser()
            return p if p.is_file() else None
        p = _default_maps_path(self.env_name)
        if p is not None and p.is_file():
            return p
        cached = _cache_maps_path(self.env_name)
        if cached.is_file():
            return cached
        return None

    def _map_pos_for(self, map_idx: int | None) -> int:
        if not self._map_index_to_key:
            return 0
        if map_idx is None:
            return 0
        if map_idx in self._maps:
            return self._map_index_to_key.index(map_idx)
        if str(map_idx) in self._maps:
            return self._map_index_to_key.index(str(map_idx))
        return 0

    def _build_inner_env(self):
        name = self.env_name
        if (
            self._current_map_key is not None
            and self._current_map_idx is not None
        ):
            name = f'{self.env_name}_{self._current_map_idx}'
        return load_environment(
            name=name,
            map_key=self._current_map_key,
            block_dist=self._block_dist,
            turns=self._turns,
            max_episode_steps=self._max_episode_steps,
            render_mode=self.render_mode,
        )

    def _build_variation_space(self) -> swm_spaces.Dict:
        n_maps = max(len(self._map_index_to_key), 1)
        init_idx = 0
        if self._current_map_idx is not None and self._map_index_to_key:
            init_idx = self._map_pos_for(self._current_map_idx)

        return swm_spaces.Dict(
            {
                'maze': swm_spaces.Dict(
                    {
                        'map_idx': swm_spaces.Discrete(
                            n_maps,
                            start=0,
                            init_value=init_idx,
                        ),
                    }
                ),
            },
            sampling_order=['maze'],
        )

    # ------------------------------------------------------------------
    # Drawer (maze_draw) – created lazily
    # ------------------------------------------------------------------

    def _get_drawer(self):
        if self._drawer is None:
            try:
                from stable_worldmodel.envs.diverse_maze.maze_draw import (
                    create_drawer,
                )

                env_id = getattr(self._env, 'name', self.env_name)
                self._drawer = create_drawer(self._env, env_id)
            except Exception:
                logger.debug(
                    'Could not create maze_draw drawer', exc_info=True
                )
        return self._drawer

    def _invalidate_drawer(self):
        self._drawer = None

    # ------------------------------------------------------------------
    # Map swapping
    # ------------------------------------------------------------------

    def _swap_to_variation_map(self, variation_pos: int) -> None:
        if not self._map_index_to_key:
            return
        variation_pos = min(variation_pos, len(self._map_index_to_key) - 1)
        real_key = self._map_index_to_key[variation_pos]
        new_map_key = self._maps[real_key]
        if new_map_key == self._current_map_key:
            return
        self._current_map_key = new_map_key
        self._current_map_idx = (
            int(real_key) if str(real_key).isdigit() else variation_pos
        )
        self._env = self._build_inner_env()
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._invalidate_drawer()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        if options.get('map_key') is not None:
            self._current_map_key = options['map_key']
            self._current_map_idx = options.get(
                'map_idx', self._current_map_idx
            )
            self._env = self._build_inner_env()
            self.action_space = self._env.action_space
            self.observation_space = self._env.observation_space
            self._invalidate_drawer()
        else:
            pos = int(self.variation_space['maze']['map_idx'].value)
            self._swap_to_variation_map(pos)

        try:
            obs = self._env.reset(seed=seed, options=options)
        except TypeError:
            try:
                obs = self._env.reset(seed=seed)
            except TypeError:
                obs = self._env.reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        # apply initial velocity prior after standard reset
        if self._qvel_prior:
            self._apply_qvel_prior()

        info = self._get_info()
        return obs, info

    def step(self, action: Any):
        result = self._step_inner(action)

        # action repeat: re-apply action (or variant) for extra physics steps
        for i in range(1, self._action_repeat):
            if self._action_repeat_mode == 'id':
                repeat_action = action
            elif self._action_repeat_mode == 'linear':
                repeat_action = action - i * (action / self._action_repeat)
            elif self._action_repeat_mode == 'null':
                repeat_action = np.zeros_like(action)
            else:
                raise ValueError(
                    f'Unknown action_repeat_mode: {self._action_repeat_mode}'
                )
            result = self._step_inner(repeat_action)

        return result

    def _step_inner(self, action: Any):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, _ = result
        else:
            obs, reward, done, step_info = result
            truncated = False
            if isinstance(step_info, dict) and step_info.get(
                'TimeLimit.truncated'
            ):
                truncated = True
            terminated = bool(done) and not truncated

        info = self._get_info()
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        try:
            obs = self._get_inner_obs()
            drawer = self._get_drawer()

            if drawer is not None:
                img_np = drawer.render_state(obs)
                img_np = np.asarray(img_np, dtype=np.uint8)
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                    img_np = np.transpose(img_np, (1, 2, 0))
                return img_np
        except Exception:
            logger.debug(
                'maze_draw render failed, falling back', exc_info=True
            )

        try:
            img = self._env.render()
        except TypeError:
            img = self._env.render(mode=self.render_mode)

        if img is not None:
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
        return img

    def close(self):
        if hasattr(self._env, 'close'):
            return self._env.close()
        return None

    # ------------------------------------------------------------------
    # Info / state helpers (stable-worldmodel conventions)
    # ------------------------------------------------------------------

    def _get_inner_obs(self) -> np.ndarray:
        """Flat (4,) state: positions and velocities from the inner point env."""
        e = self._env
        if hasattr(e, 'point_env') and hasattr(e.point_env, '_get_obs'):
            obs, _ = e.point_env._get_obs()
            return np.asarray(obs, dtype=np.float32).ravel()
        if hasattr(e, 'unwrapped') and hasattr(e.unwrapped, 'point_env'):
            obs, _ = e.unwrapped.point_env._get_obs()
            return np.asarray(obs, dtype=np.float32).ravel()
        raise AttributeError('Inner env exposes no readable point observation')

    def _get_info(self) -> dict:
        """Stable keys for ``World`` / policies: ``state`` (4,), ``goal_state`` (2,), ``proprio`` (2,)."""
        info: dict[str, Any] = {
            'env_name': self.env_name,
            'map_idx': self._current_map_idx,
        }
        state = self._get_inner_obs()
        info['state'] = np.asarray(state, dtype=np.float32).ravel()
        info['proprio'] = info['state'][:2].astype(np.float32, copy=False)
        goal = self.get_target()
        if goal is None:
            info['goal_state'] = np.zeros(2, dtype=np.float32)
        else:
            info['goal_state'] = np.asarray(goal, dtype=np.float32).reshape(
                -1
            )[:2]
        return info

    def _apply_qvel_prior(self) -> None:
        """Sample initial velocity from a prior distribution and override the
        current state.  Matches PLDM's ``pick_random_start`` behaviour."""
        state = self._get_inner_obs()
        if self._qvel_prior_type == 'uniform':
            mag = np.random.uniform(0, 5)
            angle = np.random.uniform(0, 2 * np.pi)
            qvel = np.array(
                [mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64
            )
        elif self._qvel_prior_type == 'normal':
            qvel = self._qvel_dist.rvs(size=2)
        else:
            raise ValueError(
                f'Unknown qvel_prior_type: {self._qvel_prior_type}'
            )

        new_state = np.concatenate([state[:2], qvel])
        set_point_state(self._env, new_state)

    def _set_state(self, state: np.ndarray) -> None:
        set_point_state(self._env, state)

    def _set_goal_state(self, goal_state: np.ndarray) -> None:
        g = np.asarray(goal_state[:2], dtype=np.float64)
        if hasattr(self._env, 'goal'):
            self._env.goal = g
            if hasattr(self._env, 'update_target_site_pos'):
                self._env.update_target_site_pos()
        elif hasattr(self._env, 'set_target'):
            self._env.set_target(g)

    def get_target(self):
        e = self._env
        if hasattr(e, 'goal'):
            return np.asarray(e.goal, dtype=np.float32).reshape(-1)[:2].copy()
        if hasattr(e, 'get_target'):
            return e.get_target()
        return None


def make_diverse_maze_env(env_name: str, **kwargs: Any) -> DiverseMazeEnv:
    """Factory for :class:`DiverseMazeEnv` (supports ``maps_path`` and other kwargs)."""
    return DiverseMazeEnv(env_name=env_name, **kwargs)
