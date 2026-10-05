"""
src/ds_scorer.py  --  STEP 3 of the DS-GRPO pipeline.

The FROZEN DS reward model. Mirrors the interface of a learned reward model in
RLHF: fit once offline, then called only forward, never updated.

    r_phi(x, y)        <->   DSScorer.score(trajectory)

Whiteboard statistics, from the operators fit in step 2:

    eps_c = (1/(K-1)) sum_k || x_{k+1} - T_c x_k ||^2
    eps_h = (1/(K-1)) sum_k || x_{k+1} - T_h x_k ||^2

    diff  = (eps_c - eps_h) / eps_c        <- literally on the board
    ratio = eps_h / eps_c                  <- the board's arrow, want >> 1
    r_DS  = log(eps_h / eps_c)             <- what we actually feed to GRPO

Why log(ratio) and not diff: diff is ASYMMETRIC. It saturates at exactly +1
when a trajectory is deep in the h-regime, but is unbounded negative when it
is deep in the c-regime. GRPO group-normalizes rewards, so one outlier would
dominate the whole group's advantage. log(ratio) is symmetric, well scaled,
and crosses zero at exactly the same decision boundary (eps_c == eps_h).

Usage:
    python src/ds_scorer.py --ops results/operators.npz --demo
"""

from __future__ import annotations

import argparse

import numpy as np

from fit_operators import apply_scaler, epsilon


class DSScorer:
    """Frozen dynamical-system reward model. No parameters are ever updated."""

    def __init__(self, T_c, T_h, mu=None, sd=None,
                 standardize=True, normalize_eps=True, eps_floor=1e-12):
        self.T_c = np.asarray(T_c, dtype=np.float64)
        self.T_h = np.asarray(T_h, dtype=np.float64)
        d = self.T_c.shape[0]
        self.mu = np.zeros(d) if mu is None else np.asarray(mu, dtype=np.float64)
        self.sd = np.ones(d) if sd is None else np.asarray(sd, dtype=np.float64)
        self.standardize = bool(standardize)
        self.normalize_eps = bool(normalize_eps)
        self.eps_floor = float(eps_floor)

    # ---------------- persistence ----------------

    @classmethod
    def load(cls, path: str) -> "DSScorer":
        d = np.load(path)
        return cls(
            T_c=d["T_c"], T_h=d["T_h"], mu=d["mu"], sd=d["sd"],
            standardize=bool(d["standardize"]),
            normalize_eps=bool(d["normalize_eps"]),
        )

    # ---------------- scoring ----------------

    def _prep(self, traj) -> np.ndarray:
        t = np.asarray(traj, dtype=np.float64)
        if t.ndim != 2:
            raise ValueError(f"trajectory must be (K, d); got shape {t.shape}")
        return apply_scaler(t, self.mu, self.sd) if self.standardize else t

    def residuals(self, traj) -> tuple[float, float]:
        """Returns (eps_c, eps_h)."""
        t = self._prep(traj)
        ec = max(epsilon(self.T_c, t, self.normalize_eps), self.eps_floor)
        eh = max(epsilon(self.T_h, t, self.normalize_eps), self.eps_floor)
        return ec, eh

    def score(self, traj) -> dict:
        """Full whiteboard statistics for one trajectory."""
        if len(traj) < 2:
            return {"eps_c": np.nan, "eps_h": np.nan,
                    "diff": 0.0, "ratio": 1.0, "reward": 0.0}
        ec, eh = self.residuals(traj)
        return {
            "eps_c": ec,
            "eps_h": eh,
            "diff": (ec - eh) / ec,          # whiteboard normalized differential
            "ratio": eh / ec,                # whiteboard ratio (want >> 1)
            "reward": float(np.log(eh / ec)) # symmetric reward fed to GRPO
        }

    def reward(self, traj) -> float:
        """The scalar GRPO consumes. Positive => correct-regime dynamics."""
        return self.score(traj)["reward"] if len(traj) >= 2 else 0.0

    # Same ergonomics as a learned RM:  r = scorer(trajectory)
    __call__ = reward

    def reward_batch(self, trajs) -> np.ndarray:
        return np.array([self.reward(t) for t in trajs], dtype=np.float64)

    def predicts_correct(self, traj) -> bool:
        """Whiteboard decision rule: eps_h >> eps_c  =>  correct regime."""
        return self.reward(traj) > 0.0


# ----------------------------------------------------------------------

def _demo(ops_path: str, traj_path: str) -> None:
    """Score held-out real trajectories and print the whiteboard table."""
    from collect import load_trajectories

    scorer = DSScorer.load(ops_path)
    success, crash = load_trajectories(traj_path)

    print(f"{'type':<10}{'eps_c':>12}{'eps_h':>12}{'diff':>10}{'ratio':>12}{'r_DS':>10}")
    print("-" * 66)
    for label, trajs in (("SUCCESS", success[:5]), ("CRASH", crash[:5])):
        for t in trajs:
            s = scorer.score(t)
            print(f"{label:<10}{s['eps_c']:12.5f}{s['eps_h']:12.5f}"
                  f"{s['diff']:10.3f}{s['ratio']:12.4f}{s['reward']:+10.3f}")

    r_c = scorer.reward_batch(success)
    r_h = scorer.reward_batch(crash)
    print("-" * 66)
    print(f"mean r_DS  success: {r_c.mean():+.4f}   crash: {r_h.mean():+.4f}   "
          f"separation: {r_c.mean() - r_h.mean():+.4f}")


def main():
    p = argparse.ArgumentParser(description="DS reward model (frozen scorer).")
    p.add_argument("--ops", type=str, default="results/operators.npz")
    p.add_argument("--trajs", type=str, default="results/trajectories.npz")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    print("=" * 68)
    print("STEP 3 — DS SCORER  r_DS = log(eps_h / eps_c)")
    print("=" * 68)
    if args.demo:
        _demo(args.ops, args.trajs)
    else:
        DSScorer.load(args.ops)
        print(f"Loaded frozen scorer from {args.ops}. Use --demo to see scores.")


if __name__ == "__main__":
    main()
