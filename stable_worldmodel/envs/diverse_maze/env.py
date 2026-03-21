from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from stable_worldmodel.envs.diverse_maze import ant_draw


def _default_maps_path(env_name: str) -> Path | None:
    base = Path(__file__).resolve().parent / "presaved_datasets"
    if "small" in env_name:
        return base / "5maps" / "train_maps.pt"
    if "large" in env_name:
        return base / "40maps" / "train_maps.pt"
    if "medium" in env_name or "ant_medium" in env_name:
        return base / "20maps" / "train_maps.pt"
    return None


@lru_cache(maxsize=8)
def _load_maps(path: str) -> dict:
    return torch.load(path)


def _resolve_map_key(
    env_name: str, map_key: str | None, map_idx: int | None
) -> tuple[str | None, int | None]:
    if map_key is not None:
        return map_key, map_idx

    maps_path = _default_maps_path(env_name)
    if maps_path is None or not maps_path.exists():
        return None, map_idx

    maps = _load_maps(str(maps_path))
    keys = list(maps.keys())
    if not keys:
        return None, map_idx

    if map_idx is None:
        selected_key = sorted(keys, key=lambda v: str(v))[0]
        return maps[selected_key], int(selected_key) if str(selected_key).isdigit() else map_idx

    if map_idx in maps:
        return maps[map_idx], map_idx
    if str(map_idx) in maps:
        return maps[str(map_idx)], map_idx

    selected_key = sorted(keys, key=lambda v: str(v))[0]
    return maps[selected_key], int(selected_key) if str(selected_key).isdigit() else map_idx


class DiverseMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_name: str,
        map_key: str | None = None,
        map_idx: int | None = None,
        block_dist: int | None = None,
        turns: int | None = None,
        max_episode_steps: int = 600,
        no_legs: bool = True,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()

        try:
            import d4rl  # noqa: F401
        except Exception:
            # D4RL is optional until a D4RL env is actually instantiated.
            pass

        self.env_name = env_name
        self.render_mode = render_mode

        resolved_map_key, resolved_map_idx = _resolve_map_key(
            env_name, map_key, map_idx
        )
        self.map_key = resolved_map_key
        self.map_idx = resolved_map_idx

        self._env = ant_draw.load_environment(
            name=env_name,
            map_key=resolved_map_key,
            block_dist=block_dist,
            turns=turns,
            max_episode_steps=max_episode_steps,
            no_legs=no_legs,
        )

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._max_episode_steps = max_episode_steps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        if options.get("map_key") is not None or options.get("map_idx") is not None:
            resolved_map_key, resolved_map_idx = _resolve_map_key(
                self.env_name,
                options.get("map_key"),
                options.get("map_idx"),
            )
            if resolved_map_key != self.map_key:
                self.map_key = resolved_map_key
                self.map_idx = resolved_map_idx
                self._env = ant_draw.load_environment(
                    name=self.env_name,
                    map_key=self.map_key,
                    block_dist=options.get("block_dist"),
                    turns=options.get("turns"),
                    max_episode_steps=self._max_episode_steps,
                    no_legs=options.get("no_legs", True),
                )
                self.action_space = self._env.action_space
                self.observation_space = self._env.observation_space

        try:
            obs = self._env.reset(seed=seed, options=options)
        except TypeError:
            try:
                obs = self._env.reset(seed=seed)
            except TypeError:
                obs = self._env.reset()

        if isinstance(obs, tuple) and len(obs) == 2:
            return obs

        return obs, {}

    def step(self, action: Any):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            return result

        obs, reward, done, info = result
        truncated = False
        if isinstance(info, dict) and info.get("TimeLimit.truncated"):
            truncated = True
        return obs, reward, bool(done), bool(truncated), info or {}

    def render(self):
        try:
            return self._env.render()
        except TypeError:
            return self._env.render(mode=self.render_mode)

    def close(self):
        if hasattr(self._env, "close"):
            return self._env.close()
        return None


def make_diverse_maze_env(env_name: str, **kwargs: Any) -> DiverseMazeEnv:
    return DiverseMazeEnv(env_name=env_name, **kwargs)
