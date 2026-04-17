from __future__ import annotations

import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register as gymnasium_register

from stable_worldmodel import spaces as swm_spaces

DEFAULT_VARIATIONS = ('maze.map_idx',)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@contextmanager
def _suppress_output():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def _convert_to_binary_array(map_key: str) -> np.ndarray:
    return np.array(
        [[1 if char == "#" else 0 for char in row] for row in map_key.split("\\")]
    )


def _create_custom_maze2d_env(name: str, maze_map: list, max_episode_steps: int):
    from gymnasium_robotics.envs.maze import PointMazeEnv

    class CustomMazeEnv(PointMazeEnv):
        def __init__(self, **kwargs):
            super().__init__(maze_map=maze_map, **kwargs)

        def step(self, action):
            next_state_dict, original_reward, terminated, truncated, info = super().step(action)
            reward = 1 if self._is_goal_reached(
                achieved_goal=next_state_dict["achieved_goal"],
                desired_goal=next_state_dict["desired_goal"],
            ) else 0
            info['success'] = reward > 0
            return next_state_dict, reward, terminated, truncated, info

        def _is_goal_reached(self, achieved_goal, desired_goal, goal_dist_threshold=0.5):
            distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
            return distance_to_goal < goal_dist_threshold

    with _suppress_output():
        env = CustomMazeEnv()

    return env


# ------------------------------------------------------------------
# load_environment  (maze2d only)
# ------------------------------------------------------------------

def load_environment(
    name: str,
    map_key: str | None = None,
    block_dist: int | None = None,
    turns: int | None = None,
    max_episode_steps: int = 600,
):
    maze_map = None

    if map_key is not None:
        maze_map = _convert_to_binary_array(map_key)
        env = _create_custom_maze2d_env(name, maze_map.tolist(), max_episode_steps)
    else:
        with _suppress_output():
            wrapped_env = gym.make(name)
            env = wrapped_env.unwrapped

    env.max_episode_steps = max_episode_steps
    env.name = name
    env.maze_map = maze_map
    env.map_key = map_key
    env.block_dist = block_dist
    env.turns = turns

    env.reset()
    env.step(env.action_space.sample())
    env.reset()

    return env


# ------------------------------------------------------------------
# Map catalogue helpers
# ------------------------------------------------------------------

def _default_maps_path(env_name: str) -> Path | None:
    base = Path(__file__).resolve().parent / "presaved_datasets"
    if "small" in env_name:
        return base / "5maps" / "train_maps.pt"
    if "large" in env_name:
        return base / "40maps" / "train_maps.pt"
    if "medium" in env_name:
        return base / "20maps" / "train_maps.pt"
    return None


@lru_cache(maxsize=8)
def _load_maps(path: str) -> dict:
    return torch.load(path)


def _set_maze2d_state(env, state: np.ndarray) -> None:
    qpos = np.copy(env.sim.data.qpos)
    qvel = np.copy(env.sim.data.qvel)
    qpos[:2] = state[:2]
    qvel[: len(state[2:])] = state[2:]
    env.set_state(qpos, qvel)


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

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        env_name: str,
        map_key: str | None = None,
        map_idx: int | None = None,
        block_dist: int | None = None,
        turns: int | None = None,
        max_episode_steps: int = 600,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()

        self.env_name = env_name
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._block_dist = block_dist
        self._turns = turns

        # --- map catalogue (diverse envs only) ---
        self._maps: dict = {}
        self._map_index_to_key: list = []
        if "diverse" in env_name:
            maps_path = _default_maps_path(env_name)
            if maps_path is not None and maps_path.exists():
                self._maps = _load_maps(str(maps_path))
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
            self._current_map_idx = int(real_key) if str(real_key).isdigit() else pos

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
        if self._current_map_key is not None and self._current_map_idx is not None:
            name = f"{self.env_name}_{self._current_map_idx}"
        return load_environment(
            name=name,
            map_key=self._current_map_key,
            block_dist=self._block_dist,
            turns=self._turns,
            max_episode_steps=self._max_episode_steps,
        )

    def _build_variation_space(self) -> swm_spaces.Dict:
        n_maps = max(len(self._map_index_to_key), 1)
        init_idx = 0
        if self._current_map_idx is not None and self._map_index_to_key:
            init_idx = self._map_pos_for(self._current_map_idx)

        return swm_spaces.Dict(
            {
                "maze": swm_spaces.Dict(
                    {
                        "map_idx": swm_spaces.Discrete(
                            n_maps, start=0, init_value=init_idx,
                        ),
                    }
                ),
            },
            sampling_order=["maze"],
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

                env_id = getattr(self._env, "name", self.env_name)
                self._drawer = create_drawer(self._env, env_id)
            except Exception:
                logger.debug("Could not create maze_draw drawer", exc_info=True)
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
        self._current_map_idx = int(real_key) if str(real_key).isdigit() else variation_pos
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
            self.variation_space, seed, options, DEFAULT_VARIATIONS,
        )

        if options.get("map_key") is not None:
            self._current_map_key = options["map_key"]
            self._current_map_idx = options.get("map_idx", self._current_map_idx)
            self._env = self._build_inner_env()
            self.action_space = self._env.action_space
            self.observation_space = self._env.observation_space
            self._invalidate_drawer()
        else:
            pos = int(self.variation_space["maze"]["map_idx"].value)
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

        info = self._get_info()
        return obs, info

    def step(self, action: Any):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, _ = result
        else:
            obs, reward, done, step_info = result
            truncated = False
            if isinstance(step_info, dict) and step_info.get("TimeLimit.truncated"):
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
            logger.debug("maze_draw render failed, falling back", exc_info=True)

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
        if hasattr(self._env, "close"):
            return self._env.close()
        return None

    # ------------------------------------------------------------------
    # Info / state helpers (stable-worldmodel conventions)
    # ------------------------------------------------------------------

    def _get_inner_obs(self) -> np.ndarray:
        if hasattr(self._env, "_get_obs"):
            return self._env._get_obs()
        if hasattr(self._env, "unwrapped") and hasattr(self._env.unwrapped, "_get_obs"):
            return self._env.unwrapped._get_obs()
        raise AttributeError("Inner env exposes no _get_obs()")

    def _get_info(self) -> dict:
        info: dict[str, Any] = {
            "env_name": self.env_name,
            "map_idx": self._current_map_idx,
        }
        try:
            obs = self._get_inner_obs()
            info["state"] = obs
            info["proprio"] = obs[:2]
        except Exception:
            pass
        target = self.get_target()
        if target is not None:
            info["goal_state"] = np.array(target, dtype=np.float32)
        return info

    def _set_state(self, state: np.ndarray) -> None:
        _set_maze2d_state(self._env, state)

    def _set_goal_state(self, goal_state: np.ndarray) -> None:
        if hasattr(self._env, "set_target"):
            self._env.set_target(goal_state[:2])

    def get_target(self):
        if hasattr(self._env, "get_target"):
            return self._env.get_target()
        return None


def make_diverse_maze_env(env_name: str, **kwargs: Any) -> DiverseMazeEnv:
    return DiverseMazeEnv(env_name=env_name, **kwargs)
