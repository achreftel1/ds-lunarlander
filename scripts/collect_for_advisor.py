"""
scripts/collect_for_advisor.py

Deliverable for the advisor: TWO datasets of trajectory matrices.

    results/correct_trajectories.npz   N successful landings, from the TRAINED agent
    results/wrong_trajectories.npz     N crashes,             from the RANDOM agent

Each trajectory is ONE matrix of shape (8, K_i):
    rows    = the 8 state variables  (x, y, vx, vy, angle, ang_vel, leg1, leg2)
    columns = time steps k = 1..K_i   (K_i varies per episode -- the horizon
                                        depends on the actions taken)

Only the requested regime is kept in each file:
    trained agent -> keep ONLY episodes that land   (env terminal reward +100)
    random agent  -> keep ONLY episodes that crash  (env terminal reward -100)
Everything else (trained-agent crashes, truncated hover episodes, random-agent
landings) is discarded and reported in the summary.

Usage:
    python scripts/collect_for_advisor.py --n 100                            # smoke test (~1 min)
    python scripts/collect_for_advisor.py --n 10000                          # full run, PPO agent
    python scripts/collect_for_advisor.py --n 10000 --extra-train 0          # skip extra PPO training
    python scripts/collect_for_advisor.py --n 10000 --good-agent heuristic   # fallback, no NN, ~10 min
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from collect import (
    make_env, rollout, label_episode, heuristic_action, make_random_agent,
    STATE_NAMES, PPO_MODEL_PATH,
)

RESULTS = "results"
CORRECT_PATH = os.path.join(RESULTS, "correct_trajectories.npz")
WRONG_PATH = os.path.join(RESULTS, "wrong_trajectories.npz")
README_PATH = os.path.join(RESULTS, "README_for_advisor.txt")


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------

def get_trained_policy(extra_train: int, seed: int):
    """Load the cached PPO agent (train it if missing), optionally train it
    further, save it, and return a policy function s -> action."""
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise SystemExit("pip install stable-baselines3   (or use --good-agent heuristic)") from e

    env = make_env()
    if os.path.exists(PPO_MODEL_PATH):
        print(f"Loading cached PPO agent from {PPO_MODEL_PATH}")
        model = PPO.load(PPO_MODEL_PATH, env=env)
    else:
        print("No cached PPO agent -- training from scratch (400k steps first)")
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=400_000)

    if extra_train > 0:
        print(f"Continuing PPO training for {extra_train:,} more steps "
              f"(~{extra_train / 400_000 * 8:.0f} min on CPU) ...")
        model.learn(total_timesteps=extra_train, reset_num_timesteps=False)

    os.makedirs(RESULTS, exist_ok=True)
    model.save(PPO_MODEL_PATH)
    env.close()

    def policy(s: np.ndarray) -> int:
        a, _ = model.predict(s, deterministic=True)
        return int(a)
    return policy


def evaluate_policy(policy, n_eps: int, seed: int) -> float:
    """Landing rate of a policy over n_eps fresh episodes (the 'is the agent good?' check)."""
    env = make_env()
    rng = np.random.default_rng(seed)
    counts = {"c": 0, "h": 0, None: 0}
    for _ in range(n_eps):
        _, r, trunc = rollout(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
        counts[label_episode(r, trunc)] += 1
    env.close()
    rate = counts["c"] / n_eps
    print(f"  agent check over {n_eps} episodes: "
          f"landed {counts['c']}, crashed {counts['h']}, hovered/truncated {counts[None]} "
          f"-> landing rate {rate:.0%}")
    return rate


# ----------------------------------------------------------------------
# Collection: run episodes UNTIL n trajectories of the wanted regime are kept
# ----------------------------------------------------------------------

def collect_until(policy, want: str, n: int, seed: int, tag: str):
    env = make_env()
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    counts = {"c": 0, "h": 0, None: 0}
    episodes = 0
    t0 = time.time()

    while len(kept) < n:
        traj, term_r, trunc = rollout(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
        episodes += 1
        label = label_episode(term_r, trunc)
        counts[label] += 1
        if label == want:
            kept.append(np.ascontiguousarray(traj.T, dtype=np.float32))   # (8, K_i)

        if episodes % 500 == 0 or len(kept) == n:
            elapsed = time.time() - t0
            eta = elapsed / max(len(kept), 1) * (n - len(kept))
            print(f"  [{tag}] episodes {episodes:6d} | kept {len(kept):6d}/{n} | "
                  f"landed {counts['c']} crashed {counts['h']} truncated {counts[None]} | "
                  f"{elapsed / 60:5.1f} min, ETA {eta / 60:5.1f} min")

    env.close()
    return kept, counts, episodes


def save_matrices(path: str, mats: list[np.ndarray], agent: str, regime: str) -> None:
    obj = np.empty(len(mats), dtype=object)
    for i, m in enumerate(mats):
        obj[i] = m
    lengths = np.array([m.shape[1] for m in mats], dtype=np.int32)
    np.savez_compressed(
        path,
        trajectories=obj,                       # trajectories[i] has shape (8, K_i)
        lengths=lengths,                        # K_i for each trajectory
        state_names=np.array(STATE_NAMES),      # the 8 row labels
        agent=np.array(agent),
        regime=np.array(regime),
    )


def describe(name: str, mats: list[np.ndarray], counts: dict, episodes: int) -> str:
    L = np.array([m.shape[1] for m in mats])
    return (
        f"{name}\n"
        f"  trajectories kept : {len(mats)}   each of shape (8, K_i)\n"
        f"  K_i (columns)     : mean {L.mean():.1f}, min {L.min()}, max {L.max()}\n"
        f"  episodes run      : {episodes}  "
        f"(landed {counts['c']}, crashed {counts['h']}, truncated {counts[None]})\n"
    )


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="trajectories to keep PER dataset")
    p.add_argument("--good-agent", choices=["ppo", "heuristic"], default="ppo")
    p.add_argument("--extra-train", type=int, default=1_000_000,
                   help="extra PPO training steps before collecting (ppo only)")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    report = []

    # ---------- 1) WRONG trajectories: random agent (fast, do first) ----------
    print(f"\n[1/2] RANDOM agent -> {args.n} CRASH trajectories")
    random_policy = make_random_agent(np.random.default_rng(args.seed + 1))
    wrong, w_counts, w_eps = collect_until(random_policy, "h", args.n, args.seed + 1, "random")
    save_matrices(WRONG_PATH, wrong, agent="random", regime="h (crash)")
    report.append(describe("WRONG trajectories (random agent, crashes only)", wrong, w_counts, w_eps))
    print(report[-1])

    # ---------- 2) CORRECT trajectories: trained agent ----------
    print(f"\n[2/2] TRAINED agent ({args.good_agent}) -> {args.n} LANDING trajectories")
    if args.good_agent == "ppo":
        good_policy = get_trained_policy(args.extra_train, args.seed)
        agent_name = f"PPO (stable-baselines3), model {PPO_MODEL_PATH}"
    else:
        good_policy = heuristic_action
        agent_name = "Gymnasium heuristic PD controller (not learned)"
    rate = evaluate_policy(good_policy, args.eval_episodes, args.seed + 7)
    if rate < 0.5:
        print("  WARNING: landing rate < 50%. Collection will be slow; consider more --extra-train.")

    correct, c_counts, c_eps = collect_until(good_policy, "c", args.n, args.seed + 2, "trained")
    save_matrices(CORRECT_PATH, correct, agent=agent_name, regime="c (safe landing)")
    report.append(describe(f"CORRECT trajectories ({agent_name}, landings only)", correct, c_counts, c_eps))
    print(report[-1])

    # ---------- README for the advisor ----------
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(
            "LunarLander-v3 trajectory datasets\n"
            "==================================\n\n"
            "Each trajectory is a matrix of shape (8, K_i):\n"
            "  rows    = state  x_k = (x, y, vx, vy, angle, ang_vel, leg1, leg2)\n"
            "  columns = time steps k = 1..K_i   (K_i varies: horizon depends on actions)\n\n"
            "Labels are the environment's own terminal reward:\n"
            "  +100 -> safe landing (correct), -100 -> crash (wrong).\n\n"
            f"Good-agent landing rate over {args.eval_episodes} check episodes: {rate:.0%}\n\n"
            + "\n".join(report) +
            "\nLoad in Python:\n"
            "  import numpy as np\n"
            "  d = np.load('correct_trajectories.npz', allow_pickle=True)\n"
            "  X = d['trajectories']        # object array, X[i].shape == (8, K_i)\n"
            "  K = d['lengths']             # K_i per trajectory\n"
            "  print(d['state_names'])      # row labels\n"
        )

    print(f"DONE in {(time.time() - t0) / 60:.1f} min\n")
    print("Files to send to the advisor:")
    for path in (CORRECT_PATH, WRONG_PATH, README_PATH):
        print(f"  {path}   ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
