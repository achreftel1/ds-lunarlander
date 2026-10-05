"""
src/collect.py  --  STEP 1 of the DS-GRPO pipeline.

Collects 8-dimensional state trajectories from Gymnasium's LunarLander and
labels every episode using the environment's OWN terminal reward:

    terminal reward == +100  ->  safe landing   -> regime "c" (correct)
    terminal reward == -100  ->  crash / OOB    -> regime "h" (hallucinated)

This is exact ground truth taken directly from lunar_lander.py's step():
        if self.game_over or abs(state[0]) >= 1.0:  terminated=True; reward=-100
        if not self.lander.awake:                   terminated=True; reward=+100

Truncated episodes (hit the 1000-step TimeLimit without terminating) have NO
label and are discarded.

Usage:
    python src/collect.py --episodes 600 --out results/trajectories.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "gymnasium is not installed. Run:\n"
        '    pip install swig && pip install "gymnasium[box2d]"'
    ) from e


# ----------------------------------------------------------------------
# Environment helper (handles the v3 / v2 naming difference)
# ----------------------------------------------------------------------

ENV_CANDIDATES = ("LunarLander-v3", "LunarLander-v2")

STATE_DIM = 8
N_ACTIONS = 4
STATE_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel", "leg1", "leg2"]


def make_env(render_mode: str | None = None):
    """Create LunarLander, trying v3 first then falling back to v2."""
    last_err = None
    for env_id in ENV_CANDIDATES:
        try:
            return gym.make(env_id, render_mode=render_mode)
        except Exception as err:  # NameNotFound on older/newer gymnasium
            last_err = err
    raise RuntimeError(
        f"Could not create LunarLander. Tried {ENV_CANDIDATES}. "
        f"Last error: {last_err}\n"
        'Check your install with: python -c "import gymnasium; print(gymnasium.__version__)"'
    )


# ----------------------------------------------------------------------
# Policies used ONLY for data collection (never trained)
# ----------------------------------------------------------------------

def heuristic_action(s: np.ndarray) -> int:
    """Gymnasium's own published LunarLander heuristic controller (discrete).

    Reproduced from gymnasium/envs/box2d/lunar_lander.py::heuristic.
    Lands successfully most of the time -> source of 'correct' trajectories.
    """
    angle_targ = float(np.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4))
    hover_targ = 0.55 * abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:  # a leg is touching the ground
        angle_todo = 0.0
        hover_todo = -s[3] * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2  # main engine
    if angle_todo < -0.05:
        return 3  # right orientation engine
    if angle_todo > 0.05:
        return 1  # left orientation engine
    return 0  # do nothing


def make_noisy_heuristic(epsilon: float, rng: np.random.Generator):
    """Heuristic that takes a random action with probability `epsilon`.

    This is the workhorse collector: it produces BOTH successes and crashes,
    so T_c and T_h are fit on behaviourally comparable data rather than on
    'expert vs. pure noise', which would make the separation trivially easy.
    """
    def policy(s: np.ndarray) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, N_ACTIONS))
        return heuristic_action(s)
    return policy


def make_random_policy(rng: np.random.Generator):
    def policy(s: np.ndarray) -> int:
        return int(rng.integers(0, N_ACTIONS))
    return policy


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------

def rollout(env, policy, seed: int | None = None):
    """Run one episode. Returns (trajectory, terminal_reward, truncated, env_return)."""
    state, _ = env.reset(seed=seed)
    traj = [np.asarray(state, dtype=np.float64)]
    env_return = 0.0
    last_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        traj.append(np.asarray(state, dtype=np.float64))
        env_return += float(reward)
        last_reward = float(reward)

    return np.array(traj, dtype=np.float64), last_reward, truncated, env_return


def label_episode(terminal_reward: float, truncated: bool) -> str | None:
    """'c' = safe landing, 'h' = crash, None = unlabeled (discard)."""
    if truncated:
        return None
    if terminal_reward >= 99.0:
        return "c"
    if terminal_reward <= -99.0:
        return "h"
    return None


# ----------------------------------------------------------------------
# Collection driver
# ----------------------------------------------------------------------

def collect_trajectories(
    n_episodes: int = 600,
    min_len: int = 10,
    seed: int = 0,
    verbose: bool = True,
):
    """Collect labeled trajectories from a mixture of exploratory policies."""
    rng = np.random.default_rng(seed)
    env = make_env()

    # Mixture: mostly noisy-heuristic at several noise levels (gives both
    # regimes), plus some pure random (guarantees plenty of crashes).
    schedule = [
        ("heuristic",      make_noisy_heuristic(0.00, rng), int(0.25 * n_episodes)),
        ("noisy-0.15",     make_noisy_heuristic(0.15, rng), int(0.25 * n_episodes)),
        ("noisy-0.35",     make_noisy_heuristic(0.35, rng), int(0.25 * n_episodes)),
        ("random",         make_random_policy(rng),         n_episodes - 3 * int(0.25 * n_episodes)),
    ]

    success, crash = [], []
    n_discarded = 0
    ep_counter = 0

    for name, policy, n_eps in schedule:
        n_s = n_h = 0
        for _ in range(n_eps):
            traj, term_r, trunc, _ = rollout(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
            ep_counter += 1
            label = label_episode(term_r, trunc)

            if label is None or len(traj) < min_len:
                n_discarded += 1
                continue
            if label == "c":
                success.append(traj)
                n_s += 1
            else:
                crash.append(traj)
                n_h += 1

        if verbose:
            print(f"  [{name:<12}] {n_eps:4d} episodes -> {n_s:4d} success, {n_h:4d} crash")

    env.close()

    if verbose:
        print(f"\nTOTAL: {len(success)} success (regime c), {len(crash)} crash (regime h), "
              f"{n_discarded} discarded (truncated / too short)")
        if success:
            print(f"  mean length  success: {np.mean([len(t) for t in success]):7.1f} steps")
        if crash:
            print(f"  mean length  crash  : {np.mean([len(t) for t in crash]):7.1f} steps")

    return success, crash


# ----------------------------------------------------------------------
# Serialization (flat arrays + lengths -> no pickle needed)
# ----------------------------------------------------------------------

def _flatten(trajs: list[np.ndarray]):
    if not trajs:
        return np.zeros((0, STATE_DIM)), np.zeros((0,), dtype=np.int64)
    return (np.concatenate(trajs, axis=0),
            np.array([len(t) for t in trajs], dtype=np.int64))


def _unflatten(flat: np.ndarray, lens: np.ndarray) -> list[np.ndarray]:
    out, i = [], 0
    for n in lens:
        out.append(flat[i:i + int(n)])
        i += int(n)
    return out


def save_trajectories(path: str, success: list[np.ndarray], crash: list[np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cf, cl = _flatten(success)
    hf, hl = _flatten(crash)
    np.savez_compressed(path, c_flat=cf, c_lens=cl, h_flat=hf, h_lens=hl)
    print(f"Saved -> {path}")


def load_trajectories(path: str):
    d = np.load(path)
    return (_unflatten(d["c_flat"], d["c_lens"]),
            _unflatten(d["h_flat"], d["h_lens"]))


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Collect labeled LunarLander trajectories.")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/trajectories.npz")
    args = p.parse_args()

    print("=" * 68)
    print("STEP 1 — COLLECTING LABELED TRAJECTORIES")
    print("=" * 68)

    success, crash = collect_trajectories(args.episodes, args.min_len, args.seed)

    if len(success) < 20 or len(crash) < 20:
        print("\n!! WARNING: fewer than 20 trajectories in one regime.")
        print("   Re-run with a larger --episodes (e.g. 1200) before fitting.")

    save_trajectories(args.out, success, crash)


if __name__ == "__main__":
    main()