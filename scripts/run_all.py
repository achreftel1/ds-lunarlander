"""
scripts/run_all.py  --  validates the DS reward model on LunarLander.

    Step 1  collect labeled trajectories        (collect.py)
    Step 2  fit T_c and T_h                     (fit_operators.py)
    Step 3  verify the whiteboard decision rule (ds_scorer.py)

Scope: this project validates the DS method AS A REWARD MODEL only.
Policy optimization (GRPO) is deliberately out of scope here -- it belongs
to the LLM paper, where the policy starts from an SFT-quality model.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --episodes 1200 --alpha 1e-1
    python scripts/run_all.py --skip-collect        # reuse saved trajectories
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from collect import collect_trajectories, save_trajectories, load_trajectories
from fit_operators import fit_operators, save_operators
from ds_scorer import DSScorer


RESULTS = "results"
TRAJ_PATH = os.path.join(RESULTS, "trajectories.npz")
OPS_PATH = os.path.join(RESULTS, "operators.npz")
PLOT_PATH = os.path.join(RESULTS, "separation.png")


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def plot_separation(r_c: np.ndarray, r_h: np.ndarray, path: str) -> None:
    """Histogram of r_DS for both regimes -- the one figure for the meeting."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping figure)")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(min(r_h.min(), r_c.min()), max(r_h.max(), r_c.max()), 50)
    ax.hist(r_c, bins=bins, alpha=0.65, label=f"safe landing (n={len(r_c)})", color="tab:green")
    ax.hist(r_h, bins=bins, alpha=0.65, label=f"crash (n={len(r_h)})", color="tab:red")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1,
               label="decision boundary  r_DS = 0")
    ax.set_xlabel(r"$r_{DS} = \log(\epsilon_h / \epsilon_c)$")
    ax.set_ylabel("trajectories")
    ax.set_title("DS reward model separates the two dynamical regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {path}")


def main():
    p = argparse.ArgumentParser(description="Validate the DS reward model (steps 1-3).")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-collect", action="store_true",
                   help="reuse results/trajectories.npz")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-affine", action="store_true")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()

    # ---------------- STEP 1 ----------------
    banner("STEP 1/3 — COLLECT LABELED TRAJECTORIES")
    if args.skip_collect and os.path.exists(TRAJ_PATH):
        success, crash = load_trajectories(TRAJ_PATH)
        print(f"reused {TRAJ_PATH}: {len(success)} success / {len(crash)} crash")
    else:
        success, crash = collect_trajectories(args.episodes, seed=args.seed)
        save_trajectories(TRAJ_PATH, success, crash)

    if len(success) < 20 or len(crash) < 20:
        print("\nSTOP: not enough labeled data in one regime.")
        print("  Re-run with --episodes 1500. If successes stay near zero, the")
        print("  heuristic is failing -- check your gymnasium version.")
        return 1

    # ---------------- STEP 2 ----------------
    banner("STEP 2/3 — FIT T_c AND T_h")
    ops = fit_operators(
        success, crash,
        alpha=args.alpha,
        standardize=not args.no_standardize,
        affine=not args.no_affine,
        seed=args.seed,
    )
    save_operators(OPS_PATH, ops)

    # ---------------- STEP 3 ----------------
    banner("STEP 3/3 — VALIDATE THE DS REWARD MODEL")
    scorer = DSScorer.load(OPS_PATH)
    r_c = scorer.reward_batch(success)
    r_h = scorer.reward_batch(crash)
    sep = float(r_c.mean() - r_h.mean())
    acc_all = (np.sum(r_c > 0) + np.sum(r_h <= 0)) / max(len(r_c) + len(r_h), 1)

    print(f"  mean r_DS | safe landing : {r_c.mean():+8.4f}   (want > 0)")
    print(f"  mean r_DS | crash        : {r_h.mean():+8.4f}   (want < 0)")
    print(f"  separation               : {sep:+8.4f}   (want >> 0)")
    print(f"  accuracy, all data       : {acc_all:8.1%}")
    print(f"  accuracy, HELD-OUT       : {ops['accuracy']:8.1%}   <- the number to report")

    if sep <= 0:
        print("\nFAIL: operators do not separate the regimes.")
        print("  Knobs: --no-standardize / --no-affine, --alpha 1e-1, --episodes 1500.")
        print("  If none work, a single global linear operator may underfit the")
        print("  contact-switching dynamics -> EDMD lifting (question for advisor).")
        return 1

    print("\n  PASS — the DS score separates correct from wrong dynamics on a")
    print("  real physical system, using only the whiteboard math.")

    plot_separation(r_c, r_h, PLOT_PATH)

    banner(f"VALIDATION COMPLETE in {time.time() - t0:.1f}s")
    print(f"  {TRAJ_PATH}   labeled trajectories")
    print(f"  {OPS_PATH}      fitted T_c, T_h")
    print(f"  {PLOT_PATH}     separation figure for the meeting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
