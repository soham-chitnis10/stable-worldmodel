"""Oracle expert for diverse point-maze: BFS over map grid + PD velocity control.

Mirrors HJEPA's ``NavigationWrapper.get_oracle_subgoal`` semantics while exposing
the same ``get_action(info_dict)`` surface used by other stable-worldmodel envs
(see ``stable_worldmodel.envs.two_room.expert_policy``).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Wall-aware expert.

    Behavior per sub-env:
      - Look up the current maze layout (``map_key``) and gymnasium-robotics
        ``MazeMap`` on the underlying :class:`DiverseMazeEnv`.
      - Convert agent and goal xy to grid ``(row, col)`` via
        ``maze.cell_xy_to_rowcol``. BFS on the layout gives the next grid
        cell on the shortest path.
      - Drive toward that subgoal (cell center) with a PD controller:
        ``kp * (subgoal - pos) - kd * vel``, clipped to ``[-1, 1]``.
      - If the layout or maze is unavailable (e.g. a non-diverse variant),
        fall back to pure PD toward ``goal_state``.
    """

    def __init__(
        self,
        action_noise: float = 0.0,
        kp: float = 2.5,
        kd: float = 0.8,
        action_repeat_prob: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "expert"
        self.action_noise = float(action_noise)
        self.kp = float(kp)
        self.kd = float(kd)
        self.action_repeat_prob = float(action_repeat_prob)
        self.set_seed(seed)
        self._last_action: np.ndarray | None = None

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env: Any) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "state" in info_dict, "'state' must be provided in info_dict"
        assert "goal_state" in info_dict, "'goal_state' must be provided in info_dict"

        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            envs = [e.unwrapped for e in base_env.envs]
            is_vectorized = True
        else:
            envs = [base_env]
            is_vectorized = False

        shape = self.env.action_space.shape
        actions = np.zeros(shape, dtype=np.float32)

        for i, dm_env in enumerate(envs):
            if is_vectorized:
                st = np.asarray(info_dict["state"][i], dtype=np.float32).squeeze()
                goal = np.asarray(
                    info_dict["goal_state"][i], dtype=np.float32
                ).squeeze()
            else:
                st = np.asarray(info_dict["state"], dtype=np.float32).squeeze()
                goal = np.asarray(info_dict["goal_state"], dtype=np.float32).squeeze()

            pos = st[:2]
            vel = st[2:4] if st.shape[0] >= 4 else np.zeros(2, dtype=np.float32)

            subgoal = self._resolve_subgoal(dm_env, pos, goal)
            cmd = self.kp * (subgoal - pos) - self.kd * vel
            cmd = np.clip(cmd, -1.0, 1.0).astype(np.float32)

            if self.action_noise > 0.0:
                cmd = cmd + self.rng.normal(
                    0.0, self.action_noise, size=cmd.shape
                ).astype(np.float32)
                cmd = np.clip(cmd, -1.0, 1.0)

            if is_vectorized:
                actions[i] = cmd
            else:
                actions[:] = cmd

        if self._last_action is not None and self.action_repeat_prob > 0.0:
            if is_vectorized:
                repeat_mask = (
                    self.rng.uniform(0.0, 1.0, size=(actions.shape[0],))
                    < self.action_repeat_prob
                )
                actions[repeat_mask] = self._last_action[repeat_mask]
            else:
                if self.rng.uniform(0.0, 1.0) < self.action_repeat_prob:
                    actions = self._last_action

        self._last_action = actions.copy()
        return actions.astype(np.float32)

    # ------------------------------------------------------------------
    # Oracle subgoal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_subgoal(dm_env: Any, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return next-cell xy on the shortest path, or ``goal`` if unresolvable."""
        map_key = getattr(dm_env, "_current_map_key", None)
        inner = getattr(dm_env, "_env", None)
        maze = getattr(inner, "maze", None) if inner is not None else None
        if map_key is None or maze is None:
            return goal.astype(np.float32)

        layout = map_key.split("\\")
        if not layout or not layout[0]:
            return goal.astype(np.float32)

        try:
            start_rc = tuple(
                int(v)
                for v in maze.cell_xy_to_rowcol(np.asarray(pos, dtype=np.float64))
            )
            goal_rc = tuple(
                int(v)
                for v in maze.cell_xy_to_rowcol(np.asarray(goal, dtype=np.float64))
            )
        except Exception:
            return goal.astype(np.float32)

        if start_rc == goal_rc:
            return goal.astype(np.float32)

        path = _bfs_shortest_path(layout, start_rc, goal_rc)
        if not path:
            return goal.astype(np.float32)

        next_rc = path[0]
        try:
            next_xy = maze.cell_rowcol_to_xy(np.asarray(next_rc, dtype=np.int64))
        except Exception:
            return goal.astype(np.float32)
        return np.asarray(next_xy, dtype=np.float32)


def _bfs_shortest_path(
    layout: list[str],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """BFS on a row/col grid of characters. Returns the path from ``start`` to
    ``goal`` (excluding ``start``). Empty list if no path exists."""
    height = len(layout)
    width = len(layout[0]) if height else 0

    def passable(r: int, c: int) -> bool:
        return 0 <= r < height and 0 <= c < width and layout[r][c] != "#"

    if not passable(*goal):
        return []
    if start == goal:
        return []

    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) == goal:
                return path + [(nr, nc)]
            if passable(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return []
