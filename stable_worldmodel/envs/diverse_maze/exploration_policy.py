"""Exploration policies for diverse point-maze data collection.

Two exploration strategies ported from the HJEPA/PLDM data-generation pipeline,
adapted to the stable-worldmodel ``BasePolicy.get_action(info_dict)`` API:

- :class:`UniformPolicy` -- random unit-norm actions, optionally held for
  ``resample_every`` steps.
- :class:`OUTrajectoryPolicy` -- Ornstein-Uhlenbeck noise with occasional
  velocity-aware "turn" manoeuvres that produce smoother, more realistic
  trajectories.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import truncnorm

from stable_worldmodel.policy import BasePolicy


def _sample_vector(rng: np.random.Generator, max_norm: float = 1.0) -> np.ndarray:
    magnitude = rng.uniform(0, max_norm)
    angle = rng.uniform(0, 2 * np.pi)
    return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle)], dtype=np.float32)


def _bound_vector_norm(vec: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > max_norm:
        vec = (vec / norm) * max_norm
    return vec


# ------------------------------------------------------------------
# Uniform policy
# ------------------------------------------------------------------

class UniformPolicy(BasePolicy):
    """Random 2-D actions with bounded norm, optionally held for several steps.

    Each sampled action has a random direction and a magnitude drawn
    uniformly from ``[0, max_norm]``.  The same action is repeated for
    ``resample_every`` consecutive environment steps before a new one
    is drawn.

    Parameters
    ----------
    max_norm : float
        Upper bound on the action vector norm.
    resample_every : int
        Number of consecutive steps to keep the same action.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        resample_every: int = 1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "uniform"
        self.max_norm = float(max_norm)
        self.resample_every = int(resample_every)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._step_counters: np.ndarray | None = None
        self._cached_actions: np.ndarray | None = None

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        self._step_counters = np.zeros(n, dtype=np.int64)
        self._cached_actions = np.zeros((n, 2), dtype=np.float32)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        n = getattr(self.env, "num_envs", 1)
        shape = self.env.action_space.shape
        actions = np.zeros(shape, dtype=np.float32)

        for i in range(n):
            if self._step_counters[i] % self.resample_every == 0:
                self._cached_actions[i] = _sample_vector(self.rng, self.max_norm)
            self._step_counters[i] += 1
            actions[i] = self._cached_actions[i]

        return actions


# ------------------------------------------------------------------
# Helpers for OU turn manoeuvres
# ------------------------------------------------------------------

def _sample_tapered_distribution(
    rng: np.random.Generator,
    a: float = 90,
    L: float = 180,
    sigma: float = 30,
    center: float = 180,
    size: int = 1000,
) -> np.ndarray:
    samples = rng.uniform(center - L, center + L, size)
    pdf = np.zeros(size)
    for idx, x in enumerate(samples):
        d = abs(x - center)
        if d <= a:
            pdf[idx] = 1.0
        elif d <= L:
            pdf[idx] = np.exp(-((d - a) ** 2) / (2 * sigma ** 2))
    total = pdf.sum()
    if total == 0:
        pdf[:] = 1.0 / size
    else:
        pdf /= total
    return rng.choice(samples, size=size, p=pdf)


def _sample_angle_tapered(rng: np.random.Generator, initial_angle: float) -> float:
    angles = _sample_tapered_distribution(rng, a=90, L=180, sigma=30, center=180, size=1000)
    angles = np.clip(angles, 0, 360)
    rotation = initial_angle - 180
    rotated = (angles + rotation) % 360
    return float(rng.choice(rotated))


def _sample_goal_vector(rng: np.random.Generator, current_vel: np.ndarray) -> np.ndarray:
    iv_norm = np.linalg.norm(current_vel)
    std = 3.25
    a_param = (0.0 - iv_norm) / std
    b_param = (5.0 - iv_norm) / std
    goal_norm = float(np.clip(truncnorm.rvs(a_param, b_param, loc=iv_norm, scale=std, random_state=rng), 0, 5))

    angle_deg = np.degrees(np.arctan2(current_vel[1], current_vel[0]))
    goal_angle_deg = _sample_angle_tapered(rng, float(angle_deg))
    goal_angle = np.radians(goal_angle_deg)
    return np.array([goal_norm * np.cos(goal_angle), goal_norm * np.sin(goal_angle)], dtype=np.float32)


def _generate_acceleration_sequence(
    rng: np.random.Generator,
    v_init: np.ndarray,
    v_goal: np.ndarray,
    N: int,
    sigma: float = 0.2,
    noise_level: float = 0.05,
) -> np.ndarray:
    delta_v = np.array(v_goal) - np.array(v_init)
    t = np.arange(N)
    mu = (N - 1) / 2.0
    base = np.exp(-0.5 * ((t - mu) / (sigma * N + 1e-8)) ** 2)
    integral = np.sum(base)
    scaling = delta_v / (integral + 1e-8)
    a_base = np.outer(base, scaling)
    noise = noise_level * rng.standard_normal((N, 2))
    return a_base + noise


def _generate_bi_acceleration_sequence(
    rng: np.random.Generator,
    v_init: np.ndarray,
    v_goal: np.ndarray,
    lower_N: int,
    upper_N: int,
    sigma: float = 0.2,
    noise_level: float = 0.05,
) -> np.ndarray:
    acc_N = rng.integers(max(lower_N, 1), upper_N + 1)
    decc_N = rng.integers(max(lower_N, 1), upper_N + 1)
    single_noise = noise_level / math.sqrt(2)
    acc_seq = _generate_acceleration_sequence(rng, v_init, np.zeros(2), int(acc_N), sigma, single_noise)
    decc_seq = _generate_acceleration_sequence(rng, np.zeros(2), v_goal, int(decc_N), sigma, single_noise)
    max_len = max(len(acc_seq), len(decc_seq))
    acc_seq = np.pad(acc_seq, ((0, max_len - len(acc_seq)), (0, 0)))
    decc_seq = np.pad(decc_seq, ((0, max_len - len(decc_seq)), (0, 0)))
    return acc_seq + decc_seq


# ------------------------------------------------------------------
# OU trajectory policy
# ------------------------------------------------------------------

class OUTrajectoryPolicy(BasePolicy):
    """Ornstein-Uhlenbeck exploration with occasional velocity-aware turns.

    This replicates the ``generate_ou_trajectory`` behaviour from the PLDM
    data-generation pipeline.  Between turns the action evolves according to an
    OU process; at random intervals (governed by ``turn_frequency``) a smooth
    acceleration/deceleration manoeuvre steers the agent toward a new sampled
    goal velocity.

    Parameters
    ----------
    ou_theta, ou_dt, ou_sigma : float
        OU process parameters.
    turn_frequency : int
        Expected number of steps between turns (Bernoulli ``1/turn_frequency``).
    turn_sigma, turn_noise_level : float
        Parameters for the bi-acceleration turn manoeuvre.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        ou_theta: float = 0.15,
        ou_dt: float = 1.0,
        ou_sigma: float = 0.2,
        turn_frequency: int = 20,
        turn_sigma: float = 0.2,
        turn_noise_level: float = 0.05,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "ou_trajectory"
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.ou_sigma = ou_sigma
        self.turn_frequency = turn_frequency
        self.turn_sigma = turn_sigma
        self.turn_noise_level = turn_noise_level
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._actions: dict[int, np.ndarray] = {}
        self._turn_queues: dict[int, list[np.ndarray]] = {}

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        self._actions = {i: self.rng.uniform(-0.1, 0.1, size=2).astype(np.float32) for i in range(n)}
        self._turn_queues = {i: [] for i in range(n)}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        n = getattr(self.env, "num_envs", 1)
        shape = self.env.action_space.shape
        actions = np.zeros(shape, dtype=np.float32)

        for i in range(n):
            if self._turn_queues[i]:
                action = self._turn_queues[i].pop(0)
            elif self.rng.random() < 1.0 / self.turn_frequency:
                vel = self._get_velocity(info_dict, i, n)
                v_goal = _sample_goal_vector(self.rng, vel)

                v_diff = v_goal - vel
                v_diff_norm = np.linalg.norm(v_diff)
                lower_N = max(int(v_diff_norm * 10 / 6), 1)
                upper_N = lower_N + 10

                acc_seq = _generate_bi_acceleration_sequence(
                    self.rng, vel, v_goal,
                    lower_N=lower_N, upper_N=upper_N,
                    sigma=self.turn_sigma, noise_level=self.turn_noise_level,
                )
                acc_seq = np.array([_bound_vector_norm(a) for a in acc_seq])
                self._turn_queues[i] = list(acc_seq)

                action = self._turn_queues[i].pop(0)
            else:
                mu = np.zeros(2)
                dx = (
                    self.ou_theta * (mu - self._actions[i]) * self.ou_dt
                    + self.ou_sigma * np.sqrt(self.ou_dt) * self.rng.standard_normal(2)
                )
                action = _bound_vector_norm(self._actions[i] + dx)

            action = action.astype(np.float32)
            self._actions[i] = action.copy()
            actions[i] = action

        return actions

    @staticmethod
    def _get_velocity(info_dict: dict, env_idx: int, n_envs: int) -> np.ndarray:
        state = info_dict.get("state")
        if state is None:
            return np.zeros(2, dtype=np.float32)
        st = np.asarray(state[env_idx] if n_envs > 1 else state, dtype=np.float32).squeeze()
        if st.shape[0] >= 4:
            return st[2:4].copy()
        return np.zeros(2, dtype=np.float32)
