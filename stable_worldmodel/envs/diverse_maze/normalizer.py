"""Z-score normalizer for diverse-maze observations and actions.

Field mapping vs pldm_envs:
    state      → pixel obs (C, H, W)  — mean/std per channel
    location   → (x, y)              — from dataset ``state[:2]``
    proprio_vel → (vx, vy)           — from dataset ``state[2:]``
    action     → 2D motor command
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stable_worldmodel.envs.diverse_maze.utils import PixelMapper


def _stats_key(env_name: str) -> str:
    if "diverse" in env_name:
        end = env_name.find("diverse") + len("diverse")
        return env_name[:end]
    return env_name


# Hardcoded per-channel pixel obs stats and per-field kinematics stats for known envs.
# state_mean/std: 3 values, one per RGB channel (float, not uint8).
NORM_STATS: dict[str, dict] = {
    "maze2d_large_diverse": {
        "state_mean": torch.tensor([146.5709, 120.0509, 93.3956]),
        "state_std": torch.tensor([84.9847, 45.3689, 10.3962]),
        "action_mean": torch.tensor([0.0004, -0.0022]),
        "action_std": torch.tensor([0.4095, 0.4082]),
        "location_mean": torch.tensor([4.3646, 4.2948]),
        "location_std": torch.tensor([2.3662, 2.3378]),
        "proprio_vel_mean": torch.tensor([-0.0291, -0.0461]),
        "proprio_vel_std": torch.tensor([1.4084, 1.4102]),
    }
}


class Normalizer:
    """Per-field z-score normalizer for diverse-maze datasets.

    Construct via :meth:`from_stats` (hardcoded), :meth:`build_normalizer`
    (computed from a collected dataset), or :meth:`build_id_normalizer` (no-op).
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        location_mean: torch.Tensor,
        location_std: torch.Tensor,
        proprio_vel_mean: torch.Tensor,
        proprio_vel_std: torch.Tensor,
        image_based: bool = True,
        pixel_mapper: PixelMapper | None = None,
    ) -> None:
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.location_mean = location_mean
        self.location_std = location_std
        self.proprio_vel_mean = proprio_vel_mean
        self.proprio_vel_std = proprio_vel_std
        self.image_based = image_based
        self.pixel_mapper = pixel_mapper

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_stats(cls, env_name: str) -> "Normalizer":
        """Build from hardcoded per-env statistics (no dataset required)."""
        key = _stats_key(env_name)
        if key not in NORM_STATS:
            raise KeyError(f"No hardcoded stats for '{key}'. Use build_normalizer() instead.")
        s = NORM_STATS[key]
        return cls(
            state_mean=s["state_mean"].clone(),
            state_std=s["state_std"].clone(),
            action_mean=s["action_mean"].clone(),
            action_std=s["action_std"].clone(),
            location_mean=s["location_mean"].clone(),
            location_std=s["location_std"].clone(),
            proprio_vel_mean=s["proprio_vel_mean"].clone(),
            proprio_vel_std=s["proprio_vel_std"].clone(),
            pixel_mapper=PixelMapper(env_name),
        )

    @classmethod
    def build_normalizer(cls, dataset_path: str | Path, env_name: str = "") -> "Normalizer":
        """Compute statistics from a collected dataset.

        Reads ``state`` (x, y, vx, vy) and ``action`` columns via
        ``stable_worldmodel.data.load_dataset``.  Pixel obs stats
        (``state_mean/std``) default to identity — pass ``env_name`` to get a
        ``PixelMapper`` attached to the result.
        """
        from stable_worldmodel.data import load_dataset

        ds = load_dataset(str(dataset_path))
        raw_states = ds.get_col_data("state").astype(np.float32)   # (N, 4)
        raw_actions = ds.get_col_data("action").astype(np.float32)  # (N, 2)

        states_t = torch.from_numpy(raw_states)
        actions_t = torch.from_numpy(raw_actions)

        locations = states_t[:, :2]
        proprio_vel = states_t[:, 2:]

        pixel_mapper = PixelMapper(env_name) if env_name else None

        return cls(
            state_mean=torch.zeros(3),
            state_std=torch.ones(3),
            action_mean=actions_t.mean(0),
            action_std=actions_t.std(0).clamp(min=1e-6),
            location_mean=locations.mean(0),
            location_std=locations.std(0).clamp(min=1e-6),
            proprio_vel_mean=proprio_vel.mean(0),
            proprio_vel_std=proprio_vel.std(0).clamp(min=1e-6),
            pixel_mapper=pixel_mapper,
        )

    @classmethod
    def build_id_normalizer(cls) -> "Normalizer":
        """Identity (no-op) normalizer."""
        return cls(
            state_mean=torch.zeros(3),
            state_std=torch.ones(3),
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
            location_mean=torch.zeros(2),
            location_std=torch.ones(2),
            proprio_vel_mean=torch.zeros(2),
            proprio_vel_std=torch.ones(2),
        )

    # ------------------------------------------------------------------
    # Normalize / unnormalize
    # ------------------------------------------------------------------

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize pixel obs tensor (..., C, H, W)."""
        mean = self.state_mean.view(-1, 1, 1).to(state.device)
        std = self.state_std.view(-1, 1, 1).to(state.device) + 1e-6
        ch = state.shape[-3]
        if ch < mean.shape[0] and mean.shape[0] % ch == 0:
            mean, std = mean[:ch], std[:ch]
        return (state - mean) / std

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.view(-1, 1, 1).to(state.device)
        std = self.state_std.view(-1, 1, 1).to(state.device)
        ch = state.shape[-3]
        if ch < mean.shape[0] and mean.shape[0] % ch == 0:
            mean, std = mean[:ch], std[:ch]
        return state * std + mean

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean.to(action.device)) / (self.action_std.to(action.device) + 1e-6)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std.to(action.device) + self.action_mean.to(action.device)

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (self.location_std.to(location.device) + 1e-6)

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(location.device)

    def normalize_proprio_vel(self, proprio_vel: torch.Tensor) -> torch.Tensor:
        return (proprio_vel - self.proprio_vel_mean.to(proprio_vel.device)) / (self.proprio_vel_std.to(proprio_vel.device) + 1e-6)

    def unnormalize_proprio_vel(self, proprio_vel: torch.Tensor) -> torch.Tensor:
        return proprio_vel * self.proprio_vel_std.to(proprio_vel.device) + self.proprio_vel_mean.to(proprio_vel.device)

    # ------------------------------------------------------------------
    # Device / persistence
    # ------------------------------------------------------------------

    def to(self, device) -> "Normalizer":
        for attr in (
            "state_mean", "state_std",
            "action_mean", "action_std",
            "location_mean", "location_std",
            "proprio_vel_mean", "proprio_vel_std",
        ):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "location_mean": self.location_mean,
                "location_std": self.location_std,
                "proprio_vel_mean": self.proprio_vel_mean,
                "proprio_vel_std": self.proprio_vel_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        s = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            state_mean=s["state_mean"],
            state_std=s["state_std"],
            action_mean=s["action_mean"],
            action_std=s["action_std"],
            location_mean=s["location_mean"],
            location_std=s["location_std"],
            proprio_vel_mean=s["proprio_vel_mean"],
            proprio_vel_std=s["proprio_vel_std"],
        )
