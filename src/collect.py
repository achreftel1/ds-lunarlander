"""
src/collect.py  --  STEP 1: collect labeled trajectories (advisor's spec).

Advisor's specification (meeting notes):
    * GOOD agent  (competent control)  -> mostly good landing trajectories
    * RANDOM agent (very bad control)  -> mostly bad/crash trajectories
    * ~10,000 episodes total, variable lengths (horizon depends on actions)
    * Label every episode with the env's OWN terminal reward:
          +100  ->  safe landing  -> regime "c"
          -100  ->  crash / OOB   -> regime "h"
      (a good agent can still fail, a random agent can still land --
       the LABEL always comes from the environment, never from the agent)

Ground truth from gymnasium/envs/box2d/lunar_lander.py::step():
        if self.game_over or abs(state[0]) >= 1.0:  terminated=True; reward=-100
        if not self.lander.awake:                   terminated=True; reward=+100

Truncated episodes (hit the TimeLimit without terminating) have NO label
and are discarded.

Usage:
    python src/collect.py --episodes 10000
    python src/collect.py --episodes 10000 --good-agent ppo   # needs stable-baselines3
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

PPO_MODEL_PATH = os.path.join("results", "ppo_lunarlander.zip")


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
# The two agents from the advisor's spec
# ----------------------------------------------------------------------

def heuristic_action(s: np.ndarray) -> int:
    """Gymnasium's own published LunarLander heuristic controller (discrete).

    Reproduced from gymnasium/envs/box2d/lunar_lander.py::heuristic.
    Lands ~94% of the time -> the GOOD agent (no training needed).
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


def make_random_agent(rng: np.random.Generator):
    """The BAD agent: uniformly random actions -> very bad control."""
    def policy(s: np.ndarray) -> int:
        return int(rng.integers(0, N_ACTIONS))
    return policy


def make_good_agent(kind: str, train_steps: int = 400_000):
    """GOOD agent: 'heuristic' (default, instant) or 'ppo' (trained).

    'ppo' requires stable-baselines3. The model is trained once and cached
    at results/ppo_lunarlander.zip, so subsequent runs load it instantly.
    """
    if kind == "heuristic":
        return heuristic_action

    if kind == "ppo":
        try:
            from stable_baselines3 import PPO  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit(
                "stable-baselines3 is not installed. Run:\n"
                "    pip install stable-baselines3\n"
                "or use --good-agent heuristic (no training needed)."
            ) from e

        if os.path.exists(PPO_MODEL_PATH):
            print(f"Loading cached PPO agent from {PPO_MODEL_PATH}")
            model = PPO.load(PPO_MODEL_PATH)
        else:
            print(f"Training PPO agent for {train_steps} steps "
                  f"(one-time, ~20-40 min on CPU)...")
            env = make_env()
            model = PPO("MlpPolicy", env, verbose=1, seed=0)
            model.learn(total_timesteps=train_steps)
            os.makedirs("results", exist_ok=True)
            model.save(PPO_MODEL_PATH)
            env.close()
            print(f"Saved trained agent -> {PPO_MODEL_PATH}")

        def policy(s: np.ndarray) -> int:
            action, _ = model.predict(s, deterministic=True)
            return int(action)
        return policy

    raise ValueError(f"unknown good agent kind: {kind!r}")


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------

def rollout(env, policy, seed: int | None = None):
    """Run one episode. Returns (trajectory, terminal_reward, truncated).

    Trajectory length is VARIABLE -- the horizon depends on the actions
    taken (crashes ~100 steps, landings ~300). This is fine: the operator
    fit stacks individual (x_k -> x_{k+1}) transitions, so trajectories
    of different lengths contribute naturally.
    """
    state, _ = env.reset(seed=seed)
    traj = [np.asarray(state, dtype=np.float32)]
    last_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        traj.append(np.asarray(state, dtype=np.float32))
        last_reward = float(reward)

    return np.array(traj, dtype=np.float32), last_reward, truncated


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
# Collection driver (advisor's spec: good agent + random agent, 10k total)
# ----------------------------------------------------------------------

def collect_trajectories(
    n_episodes: int = 10_000,
    good_frac: float = 0.5,
    good_agent: str = "heuristic",
    min_len: int = 10,
    seed: int = 0,
    verbose: bool = True,
):
    """Collect labeled trajectories from the GOOD and RANDOM agents.

    Returns (success, crash) lists of (K_i, 8) float32 arrays.
    The label is ALWAYS the environment's terminal reward -- a good-agent
    episode that crashes goes into `crash`, and vice versa.
    """
    rng = np.random.default_rng(seed)
    env = make_env()

    n_good = int(good_frac * n_episodes)
    schedule = [
        ("good",   make_good_agent(good_agent), n_good),
        ("random", make_random_agent(rng),      n_episodes - n_good),
    ]

    success, crash = [], []
    n_discarded = 0

    for name, policy, n_eps in schedule:
        n_s = n_h = 0
        for i in range(n_eps):
            traj, term_r, trunc = rollout(env, policy,
                                          seed=int(rng.integers(0, 2**31 - 1)))
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

            if verbose and (i + 1) % 500 == 0:
                print(f"  [{name:<7}] {i + 1:5d}/{n_eps} episodes "
                      f"({n_s} success, {n_h} crash so far)")

        if verbose:
            print(f"  [{name:<7}] DONE: {n_eps:5d} episodes -> "
                  f"{n_s:5d} success, {n_h:5d} crash")

    env.close()

    if verbose:
        print(f"\nTOTAL: {len(success)} success (regime c), "
              f"{len(crash)} crash (regime h), {n_discarded} discarded")
        if success:
            lens = [len(t) for t in success]
            print(f"  success length: mean {np.mean(lens):6.1f}  "
                  f"min {min(lens)}  max {max(lens)}")
        if crash:
            lens = [len(t) for t in crash]
            print(f"  crash   length: mean {np.mean(lens):6.1f}  "
                  f"min {min(lens)}  max {max(lens)}")

    return success, crash


# ----------------------------------------------------------------------
# Serialization (flat arrays + lengths -> no pickle needed)
# ----------------------------------------------------------------------

def _flatten(trajs: list[np.ndarray]):
    if not trajs:
        return np.zeros((0, STATE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (np.concatenate(trajs, axis=0).astype(np.float32),
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
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved -> {path}  ({size_mb:.1f} MB)")


def load_trajectories(path: str):
    d = np.load(path)
    success = _unflatten(d["c_flat"], d["c_lens"])
    crash = _unflatten(d["h_flat"], d["h_lens"])
    return success, crash


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Step 1: collect labeled trajectories.")
    p.add_argument("--episodes", type=int, default=10_000,
                   help="total episodes across both agents (advisor: 10000)")
    p.add_argument("--good-frac", type=float, default=0.5,
                   help="fraction of episodes run by the good agent")
    p.add_argument("--good-agent", choices=["heuristic", "ppo"], default="heuristic",
                   help="'heuristic' = Gymnasium's controller (instant); "
                        "'ppo' = train/load a PPO agent (needs stable-baselines3)")
    p.add_argument("--out", type=str, default="results/trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 68)
    print("STEP 1 — COLLECT LABELED TRAJECTORIES (good agent vs. random agent)")
    print("=" * 68)
    success, crash = collect_trajectories(
        n_episodes=args.episodes,
        good_frac=args.good_frac,
        good_agent=args.good_agent,
        seed=args.seed,
    )
    save_trajectories(args.out, success, crash)


if __name__ == "__main__":
    main()
