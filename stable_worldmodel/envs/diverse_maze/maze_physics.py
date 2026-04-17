"""Point-maze physics factory (gymnasium-robotics). No ant / OGBench here."""

from __future__ import annotations

import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import cast

import gymnasium as gym
import numpy as np


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


def convert_to_binary_array(map_key: str) -> list:
    """Parse an HJEPA ``map_key`` layout into a gymnasium-robotics ``maze_map``.

    Preserves the symbolic markers ``r`` (reset), ``g`` (goal) and ``c``
    (combined) so gymnasium-robotics enforces spawn / goal locations. Any
    other non-wall character (``O``, ``0``, spaces) becomes ``0`` (empty).
    """
    rows = map_key.split("\\")
    out: list[list] = []
    for row in rows:
        parsed: list = []
        for char in row:
            if char == "#":
                parsed.append(1)
            elif char in ("r", "g", "c"):
                parsed.append(char)
            else:
                parsed.append(0)
        out.append(parsed)
    return out


def _top_down_camera_config(name: str) -> dict:
    """Top-down camera overrides to match HJEPA-style rendering."""
    cfg: dict = {
        "elevation": -90.0,
        "azimuth": 90.0,
        "lookat": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }
    return cfg


def _apply_top_down_camera(env, name: str) -> None:
    """Install a top-down camera on the inner ``PointEnv`` MuJoCo renderer.

    Gymnasium-robotics sets a perspective ``default_cam_config`` inside
    ``PointMazeEnv.__init__``. We override it post-init and purge cached
    viewers so the next render rebuilds with the new config.
    """
    renderer = getattr(getattr(env, "point_env", None), "mujoco_renderer", None)
    if renderer is None:
        return
    merged = dict(renderer.default_cam_config or {})
    merged.update(_top_down_camera_config(name))
    renderer.default_cam_config = merged
    try:
        renderer.close()
    except Exception:
        pass
    renderer._viewers = {}


def create_custom_maze2d_env(
    name: str,
    maze_map: list,
    max_episode_steps: int,
    render_mode: str | None = "rgb_array",
):
    from gymnasium_robotics.envs.maze import PointMazeEnv

    class CustomMazeEnv(PointMazeEnv):
        def __init__(self, **kwargs):
            super().__init__(
                maze_map=maze_map,
                render_mode=render_mode,
                **kwargs,
            )

        def step(self, action):
            next_state_dict, _r, terminated, truncated, info = super().step(action)
            reward = 1 if self._is_goal_reached(
                achieved_goal=next_state_dict["achieved_goal"],
                desired_goal=next_state_dict["desired_goal"],
            ) else 0
            info["success"] = reward > 0
            return next_state_dict, reward, terminated, truncated, info

        def _is_goal_reached(self, achieved_goal, desired_goal, goal_dist_threshold=0.5):
            distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
            return distance_to_goal < goal_dist_threshold

    with suppress_output():
        env = CustomMazeEnv()

    _apply_top_down_camera(env, name)
    return env


def load_environment(
    name: str,
    map_key: str | None = None,
    block_dist: int | None = None,
    turns: int | None = None,
    max_episode_steps: int = 600,
    render_mode: str | None = "rgb_array",
):
    """Build or load a **point** maze env. Ant mazes are not supported in this module."""
    if map_key is not None and "ant" in name.lower():
        raise ValueError(
            "Ant maze loading was removed from maze_physics; use scripts under "
            "scripts/diverse_maze/ if you still need legacy ant data generation."
        )

    maze_map = None

    if map_key is not None:
        maze_map = convert_to_binary_array(map_key)
        env = create_custom_maze2d_env(
            name, maze_map, max_episode_steps, render_mode=render_mode
        )
    else:
        with suppress_output():
            wrapped_env = gym.make(name, render_mode=render_mode)
            env = cast(gym.Env, wrapped_env.unwrapped)
        _apply_top_down_camera(env, name)

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


def set_point_state(env, state: np.ndarray) -> None:
    """Set point-mass (x, y, vx, vy) on the inner MuJoCo env (same as DiverseMazeEnv._set_state)."""
    inner = env
    while True:
        if getattr(inner, "point_env", None) is not None:
            inner = inner.point_env
            break
        if hasattr(inner, "data") and hasattr(inner.data, "qpos"):
            break
        if hasattr(inner, "unwrapped") and inner.unwrapped is not inner:
            inner = inner.unwrapped
            continue
        if hasattr(inner, "env"):
            inner = inner.env
            continue
        raise TypeError(f"Could not resolve point-maze inner env from {type(env)!r}")
    qpos = np.copy(inner.data.qpos)
    qvel = np.copy(inner.data.qvel)
    qpos[:2] = state[:2]
    n_vel = min(max(len(state) - 2, 0), len(qvel))
    if n_vel:
        qvel[:n_vel] = state[2 : 2 + n_vel]
    inner.set_state(qpos, qvel)
