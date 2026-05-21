"""Tests for ``swm/maze2d_*_diverse`` (gymnasium-robotics point maze)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gymnasium_robotics = pytest.importorskip(
    "gymnasium_robotics", reason="gymnasium_robotics required for diverse maze tests"
)

import gymnasium as gym

import stable_worldmodel.envs  # noqa: F401 — register swm/* envs


def test_import_diverse_maze_package():
    import stable_worldmodel.envs.diverse_maze as dm

    assert hasattr(dm, "DiverseMazeEnv")
    assert hasattr(dm, "UniformPolicy")
    assert hasattr(dm, "OUTrajectoryPolicy")
    assert not hasattr(dm, "ExpertPolicy")


def test_import_maze_draw_no_d4rl_dependency():
    import stable_worldmodel.envs.diverse_maze.maze_draw as md

    assert hasattr(md, "create_drawer")
    assert hasattr(md, "GymnasiumPointMazeDrawer")


def test_wrappers_module_removed():
    """Legacy classic-gym wrappers are gone; importing must fail."""
    with pytest.raises(ImportError):
        import stable_worldmodel.envs.diverse_maze.wrappers  # noqa: F401


def test_no_configs_dir():
    """``envs/diverse_maze/configs/`` is not wired into SWM Hydra; deleted."""
    import stable_worldmodel.envs.diverse_maze as dm

    pkg_root = Path(dm.__file__).resolve().parent
    assert not (pkg_root / "configs").exists()


def test_scripts_diverse_maze_removed():
    """Legacy d4rl/classic-gym scripts removed; canonical path is scripts/data/collect_maze.py."""
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "scripts" / "diverse_maze").exists()
    assert (repo_root / "scripts" / "data" / "collect_maze.py").exists()


def test_maze_stats_only_exposes_render_stats():
    """STATS training-schedule dict was dead code; only RENDER_STATS remains."""
    from stable_worldmodel.envs.diverse_maze import maze_stats

    assert hasattr(maze_stats, "RENDER_STATS")
    assert not hasattr(maze_stats, "STATS")


def test_diverse_maze_render_after_reset(large_diverse_env):
    large_diverse_env.reset(seed=0)
    img = large_diverse_env.render()
    arr = np.asarray(img)
    assert arr.ndim == 3
    assert arr.shape[-1] == 3
    assert arr.dtype == np.uint8
    # Regression guard: paper-scale output, not the raw 480x480 perspective
    # frame that gymnasium-robotics emits by default.
    assert arr.shape[:2] == (98, 98), f"expected (98, 98, 3) got {arr.shape}"


@pytest.mark.parametrize(
    "env_id,expected_hw",
    [
        ("swm/maze2d_small_diverse", (64, 64)),
        ("swm/maze2d_medium_diverse", (81, 81)),
        ("swm/maze2d_large_diverse", (98, 98)),
    ],
)
def test_render_shape_per_env(env_id, expected_hw):
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        img = np.asarray(env.render())
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[-1] == 3
        assert img.shape[:2] == expected_hw, (
            f"{env_id}: expected {expected_hw}, got {img.shape[:2]}"
        )
    finally:
        env.close()


def test_render_shows_agent_and_walls(large_diverse_env):
    """Top-down render must include the red agent sphere (non-uniform frame).

    Guards against regressing to a blank/black frame or to the raw 480x480
    perspective frame (which would fail `test_render_shape_per_env` anyway).
    """
    large_diverse_env.reset(seed=0)
    img = np.asarray(large_diverse_env.render())
    # Agent sphere should be red-dominant somewhere in the frame.
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red_mask = (r > 150) & (g < 120) & (b < 120)
    assert int(red_mask.sum()) > 0, "agent (red sphere) not visible in render"
    # Wall blocks should contribute some variance (not a uniform image).
    assert img.std() > 10.0, "render appears uniform / blank"


@pytest.fixture
def two_map_catalog(tmp_path):
    import torch

    m0 = "#####\\#rOg#\\#OOO#\\#OOO#\\#####"
    m1 = "#####\\#OOg#\\#OOr#\\#OOO#\\#####"
    p = tmp_path / "train_maps.pt"
    torch.save({0: m0, 1: m1}, p)
    return p


def test_diverse_maze_loads_fixture_maps(two_map_catalog):
    from stable_worldmodel.envs.diverse_maze.env import make_diverse_maze_env

    env = make_diverse_maze_env(
        "maze2d_large_diverse",
        maps_path=str(two_map_catalog),
        render_mode="rgb_array",
    )
    try:
        assert len(env._map_index_to_key) == 2
        _obs, info = env.reset(seed=0)
        assert info["map_idx"] in (0, 1)
    finally:
        env.close()


def test_gym_make_accepts_maps_path(two_map_catalog):
    env = gym.make(
        "swm/maze2d_large_diverse",
        render_mode="rgb_array",
        maps_path=str(two_map_catalog),
    )
    try:
        assert len(env.unwrapped._maps) == 2
    finally:
        env.close()


def test_variation_map_idx_swaps_layout(two_map_catalog):
    from stable_worldmodel.envs.diverse_maze.env import make_diverse_maze_env

    env = make_diverse_maze_env(
        "maze2d_large_diverse",
        maps_path=str(two_map_catalog),
        render_mode="rgb_array",
    )
    try:
        _obs0, i0 = env.reset(
            seed=0,
            options={"variation_values": {"maze.map_idx": 0}},
        )
        _obs1, i1 = env.reset(
            seed=0,
            options={"variation_values": {"maze.map_idx": 1}},
        )
        assert i0["map_idx"] == 0
        assert i1["map_idx"] == 1
        m = env._maps
        assert m[0] != m[1]
    finally:
        env.close()


@pytest.fixture
def large_diverse_env():
    env = gym.make("swm/maze2d_large_diverse", render_mode="rgb_array")
    yield env
    env.close()


def test_diverse_maze_reset_info_has_state_goal_map_idx(large_diverse_env):
    obs, info = large_diverse_env.reset(seed=0)
    assert isinstance(obs, dict)
    assert "state" in info
    assert "goal_state" in info
    assert "proprio" in info
    assert "map_idx" in info
    assert "env_name" in info
    assert info["env_name"] == "maze2d_large_diverse"
    s = np.asarray(info["state"])
    assert s.shape == (4,)
    assert np.all(np.isfinite(s))
    g = np.asarray(info["goal_state"])
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))
    p = np.asarray(info["proprio"])
    assert p.shape == (2,)
    np.testing.assert_array_equal(p, s[:2])


def test_diverse_maze_step_info_consistent(large_diverse_env):
    large_diverse_env.reset(seed=0)
    a = large_diverse_env.action_space.sample()
    obs, _r, term, trunc, info = large_diverse_env.step(a)
    assert isinstance(obs, dict)
    assert "state" in info
    assert np.asarray(info["state"]).shape == (4,)
    assert np.all(np.isfinite(info["state"]))
    assert np.asarray(info["goal_state"]).shape == (2,)


@pytest.mark.parametrize(
    "env_id,base_name",
    [
        ("swm/maze2d_small_diverse", "maze2d_small_diverse"),
        ("swm/maze2d_medium_diverse", "maze2d_medium_diverse"),
        ("swm/maze2d_large_diverse", "maze2d_large_diverse"),
    ],
)
def test_all_registered_diverse_maze_ids_make(env_id, base_name, two_map_catalog):
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        maps_path=str(two_map_catalog),
    )
    try:
        _obs, info = env.reset(seed=0)
        assert info["env_name"] == base_name
    finally:
        env.close()


@pytest.mark.slow
def test_collect_maze_script_smoke(tmp_path, two_map_catalog):
    pytest.importorskip("hydra", reason="hydra required for collect_maze")
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "data" / "collect_maze.py"
    maps = two_map_catalog.as_posix()
    cache = tmp_path.as_posix()
    cmd = [
        sys.executable,
        str(script),
        "env_name=swm/maze2d_large_diverse",
        f"world.maps_path={maps}",
        f"cache_dir={cache}",
        "dataset_name=_pytest_collect_smoke",
        "num_traj=1",
        "world.num_envs=1",
        "world.max_episode_steps=8",
        "policy=random",
        "seed=0",
    ]
    env = dict(os.environ)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    subprocess.run(cmd, cwd=str(root), check=True, env=env)


