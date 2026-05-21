"""Procedural maze-layout generator via cellular automata.

Generates diverse connected maze grids with controllable sparsity, path length,
and wall/space constraints.  Produces a ``{idx: map_key}`` dict suitable for
:class:`DiverseMazeEnv` and saveable with ``torch.save``.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
from tqdm import tqdm


class MapGenerator:
    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        num_maps: int = 10,
        sparsity_low: float = 53,
        sparsity_high: float = 88,
        max_path_len: int = 13,
        exclude_maps: Optional[dict] = None,
        wall_coords: Optional[list] = None,
        space_coords: Optional[list] = None,
    ):
        """
        Args:
            width: Interior grid width (without border walls).
            height: Interior grid height (without border walls).
            num_maps: Number of maps to generate.
            sparsity_low: Minimum open-space percentage.
            sparsity_high: Maximum open-space percentage.
            max_path_len: Reject maps whose longest BFS distance >= this.
            exclude_maps: Dict of map keys to exclude (e.g. training maps
                when generating probe maps).
            wall_coords: List of (row, col) that must be walls.
            space_coords: List of (row, col) that must be open spaces.
        """
        self.width = width
        self.height = height
        self.num_maps = num_maps
        self.max_path_len = max_path_len
        self.sparsity_low = sparsity_low
        self.sparsity_high = sparsity_high
        self.wall_coords = wall_coords or []
        self.space_coords = space_coords or []
        self.exclude_maps = exclude_maps if exclude_maps is not None else {}

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------

    def _bfs_longest_path(self, grid, start_row, start_col):
        rows, cols = grid.shape
        visited = np.full((rows, cols), False)
        queue = deque([(start_row, start_col, 0)])
        visited[start_row, start_col] = True
        max_distance = 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pass  # just defining the pattern

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c, dist = queue.popleft()
            max_distance = max(max_distance, dist)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] == "O"
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))

        return max_distance

    def _find_longest_connected_distance(self, grid):
        rows, cols = grid.shape
        max_distance = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == "O":
                    distance = self._bfs_longest_path(grid, r, c)
                    max_distance = max(max_distance, distance)
        return max_distance

    # ------------------------------------------------------------------
    # Grid generation (cellular automata)
    # ------------------------------------------------------------------

    def _initialize_grid(self, width, height, border_fill_prob=0.5, interior_fill_prob=0.5):
        grid = np.full((height, width), "#", dtype=str)
        for r in range(height):
            for c in range(width):
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    grid[r, c] = "O" if random.random() < border_fill_prob else "#"
                else:
                    grid[r, c] = "O" if random.random() < interior_fill_prob else "#"
        return grid

    def _is_connected(self, grid):
        width, height = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        def bfs(start_x, start_y):
            queue = deque([(start_x, start_y)])
            visited[start_x, start_y] = True
            count = 1
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not visited[nx, ny]
                        and grid[nx, ny] == "O"
                    ):
                        visited[nx, ny] = True
                        queue.append((nx, ny))
                        count += 1
            return count

        for x in range(width):
            for y in range(height):
                if grid[x, y] == "O":
                    connected_count = bfs(x, y)
                    total_o_count = np.sum(grid == "O")
                    return connected_count == total_o_count
        return True

    def _open_space_to_wall(self, grid, N):
        new_grid = np.copy(grid)
        rows, cols = grid.shape
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == "O":
                    open_space_count = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == "O":
                            open_space_count += 1
                    if open_space_count > N:
                        new_grid[r, c] = "#"
        return new_grid

    def _apply_cellular_automata(self, grid, N=6):
        return self._open_space_to_wall(grid, N=N)

    def _generate_map(self, width, height, iterations=2):
        grid = self._initialize_grid(width, height)
        for _ in range(iterations):
            grid = self._apply_cellular_automata(grid)
        return grid

    def _calculate_o_percentage(self, grid):
        grid = np.array(grid)
        return (np.sum(grid == "O") / grid.size) * 100

    def _add_walls(self, grid):
        rows, cols = grid.shape
        new_array = np.full((rows + 2, cols + 2), "#", dtype=grid.dtype)
        new_array[1:-1, 1:-1] = grid
        return new_array

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def print_grid(self, grid):
        for row in grid:
            print("".join(row))

    def print_grid_from_key(self, key):
        rows = key.split("\\")
        for row in rows:
            print(row)
        print()

    # ------------------------------------------------------------------
    # Key encoding
    # ------------------------------------------------------------------

    def _generate_key(self, grid):
        return "\\".join("".join(row) for row in grid)

    # ------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------

    def _pass_wall_constraint(self, grid):
        for coord in self.wall_coords:
            if grid[coord[0], coord[1]] == "O":
                return False
        return True

    def _pass_space_constraint(self, grid):
        for coord in self.space_coords:
            if grid[coord[0], coord[1]] == "#":
                return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_diverse_maps(self) -> dict[int, str]:
        """Generate ``num_maps`` valid, unique maze layouts.

        Returns:
            ``{0: map_key_0, 1: map_key_1, ...}`` dict suitable for
            ``torch.save`` and consumption by :class:`DiverseMazeEnv`.
        """
        map_dict: dict[str, bool] = {}

        print("Generating map layouts")
        for _ in tqdm(range(self.num_maps)):
            while True:
                map_grid = self._generate_map(self.width - 2, self.height - 2)
                sparsity = self._calculate_o_percentage(map_grid)

                if not self._pass_wall_constraint(map_grid):
                    continue
                if not self._pass_space_constraint(map_grid):
                    continue
                if not (self.sparsity_low <= sparsity <= self.sparsity_high):
                    continue

                map_grid = self._add_walls(map_grid)
                key = self._generate_key(map_grid)

                if key in self.exclude_maps or key in map_dict:
                    continue
                if not self._is_connected(map_grid):
                    continue

                longest_dist = self._find_longest_connected_distance(map_grid)
                if longest_dist >= self.max_path_len:
                    continue

                map_dict[key] = True
                break

        return {i: key for i, key in enumerate(map_dict.keys())}
