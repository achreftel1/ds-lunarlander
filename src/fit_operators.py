"""
src/fit_operators.py  --  STEP 2 of the DS-GRPO pipeline.

Implements the DS explication math EXACTLY:

        x^c_{k+1} = T_c x^c_k          (correct / safe-landing regime)
        x^h_{k+1} = T_h x^h_k          (hallucinated / crash regime)

Each operator is fit by closed-form ridge least squares over all (x_k -> x_{k+1})
pairs stacked across every trajectory in that regime:

        T = argmin_T  sum_k || x_{k+1} - T x_k ||^2 + alpha ||T||_F^2
          = (Y^T X)(X^T X + alpha I)^{-1}

Two engineering additions over the raw board (both switchable, both reported):

  * standardize : LunarLander's 8 dims have wildly different scales
                  (positions ~O(1) floats, leg flags are binary 0/1).
                  Unstandardized least squares is dominated by large dims.
  * affine      : append a constant 1 to the state so the operator can learn
                  x_{k+1} = A x_k + b. Gravity is a CONSTANT acceleration, so a
                  purely linear T literally cannot represent free fall. This is
                  a legitimate and important correction -- raise it with your
                  advisor.

Usage:
    python src/fit_operators.py --in results/trajectories.npz --out results/operators.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from collect import load_trajectories, STATE_NAMES


# ----------------------------------------------------------------------
# Standardization
# ----------------------------------------------------------------------

def compute_scaler(trajectories: list[np.ndarray]):
    """Mean/std over every state visited in BOTH regimes (fit once, shared)."""
    X = np.concatenate(trajectories, axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0          # guard constant dimensions
    return mu, sd


def apply_scaler(traj: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (traj - mu) / sd


# ----------------------------------------------------------------------
# Core fit  (the whiteboard equation)
# ----------------------------------------------------------------------

def _design_matrices(trajectories: list[np.ndarray], affine: bool):
    """Stack all consecutive pairs. X holds x_k, Y holds x_{k+1}."""
    X = np.concatenate([t[:-1] for t in trajectories], axis=0)
    Y = np.concatenate([t[1:] for t in trajectories], axis=0)
    if affine:
        X = np.hstack([X, np.ones((X.shape[0], 1))])   # [x_k ; 1]
    return X, Y


def fit_T(trajectories: list[np.ndarray], alpha: float = 1e-2, affine: bool = True) -> np.ndarray:
    """Closed-form ridge solution. Returns T with shape (d, d) or (d, d+1) if affine."""
    X, Y = _design_matrices(trajectories, affine)
    p = X.shape[1]
    A = X.T @ X + alpha * np.eye(p)
    B = X.T @ Y
    return np.linalg.solve(A, B).T                      # so that Y ~= X @ T.T


def predict(T: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """One-step-ahead predictions of x_2..x_K from x_1..x_{K-1}."""
    X = traj[:-1]
    if T.shape[1] == X.shape[1] + 1:                    # affine operator
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    return X @ T.T


def epsilon(T: np.ndarray, traj: np.ndarray, normalize: bool = True) -> float:
    """Whiteboard residual  eps = sum_k || x_{k+1} - T x_k ||^2.

    normalize=True divides by (K-1), i.e. MEAN squared error per transition.
    This matters: successful landings are systematically LONGER than crashes,
    so an unnormalized SUM would conflate 'wrong regime' with 'long episode'.
    """
    resid = traj[1:] - predict(T, traj)
    sq = float(np.sum(resid ** 2))
    return sq / max(len(traj) - 1, 1) if normalize else sq


# ----------------------------------------------------------------------
# Fit both regimes + evaluate separation
# ----------------------------------------------------------------------

def train_val_split(trajs: list[np.ndarray], val_frac: float, rng: np.random.Generator):
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(val_frac * len(trajs)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return [trajs[i] for i in tr_idx], [trajs[i] for i in val_idx]


def separation_report(T_c, T_h, c_val, h_val, mu, sd, standardize, normalize):
    """Checks the whiteboard decision rule on HELD-OUT trajectories."""
    def score(traj):
        t = apply_scaler(traj, mu, sd) if standardize else traj
        ec = max(epsilon(T_c, t, normalize), 1e-12)
        eh = max(epsilon(T_h, t, normalize), 1e-12)
        return ec, eh, float(np.log(eh / ec))

    c_scores = [score(t)[2] for t in c_val]
    h_scores = [score(t)[2] for t in h_val]

    c_mean = float(np.mean(c_scores)) if c_scores else float("nan")
    h_mean = float(np.mean(h_scores)) if h_scores else float("nan")

    # Accuracy of the board's rule "r_DS > 0  =>  correct regime"
    correct = sum(s > 0 for s in c_scores) + sum(s <= 0 for s in h_scores)
    acc = correct / max(len(c_scores) + len(h_scores), 1)

    print("\n--- HELD-OUT SEPARATION (the result that matters) ---")
    print(f"  mean r_DS on SAFE-LANDING trajectories : {c_mean:+8.4f}   (want > 0)")
    print(f"  mean r_DS on CRASH        trajectories : {h_mean:+8.4f}   (want < 0)")
    print(f"  SEPARATION (c_mean - h_mean)           : {c_mean - h_mean:+8.4f}   (want >> 0)")
    print(f"  accuracy of rule  sign(r_DS)           : {acc:8.1%}")
    if c_mean - h_mean > 0:
        print("  => PASS: the whiteboard decision rule holds on real physical dynamics.")
    else:
        print("  => FAIL: operators do not separate. See 'knobs to try' in README.")
    return {"c_mean": c_mean, "h_mean": h_mean, "separation": c_mean - h_mean, "accuracy": acc}


def fit_operators(
    success: list[np.ndarray],
    crash: list[np.ndarray],
    alpha: float = 1e-2,
    standardize: bool = True,
    affine: bool = True,
    normalize_eps: bool = True,
    val_frac: float = 0.25,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    if len(success) < 5 or len(crash) < 5:
        raise ValueError(
            f"Need >=5 trajectories per regime (got c={len(success)}, h={len(crash)}). "
            "Re-run collect.py with more --episodes."
        )

    c_tr, c_val = train_val_split(success, val_frac, rng)
    h_tr, h_val = train_val_split(crash, val_frac, rng)

    mu, sd = compute_scaler(c_tr + h_tr)
    if not standardize:
        mu, sd = np.zeros_like(mu), np.ones_like(sd)

    c_fit = [apply_scaler(t, mu, sd) for t in c_tr] if standardize else c_tr
    h_fit = [apply_scaler(t, mu, sd) for t in h_tr] if standardize else h_tr

    T_c = fit_T(c_fit, alpha, affine)
    T_h = fit_T(h_fit, alpha, affine)

    print(f"\nFitted T_c  shape={T_c.shape}  on {len(c_tr)} trajectories "
          f"({sum(len(t) - 1 for t in c_tr)} transitions)")
    print(f"Fitted T_h  shape={T_h.shape}  on {len(h_tr)} trajectories "
          f"({sum(len(t) - 1 for t in h_tr)} transitions)")

    # Spectral radius: >1 means the regime's dynamics are locally unstable.
    A_c = T_c[:, :T_c.shape[0]]
    A_h = T_h[:, :T_h.shape[0]]
    rho_c = float(np.max(np.abs(np.linalg.eigvals(A_c))))
    rho_h = float(np.max(np.abs(np.linalg.eigvals(A_h))))
    print(f"  spectral radius  rho(T_c) = {rho_c:.4f}   rho(T_h) = {rho_h:.4f}")
    print("  (rho > 1 => divergent regime; a nice physical talking point for your advisor)")

    report = separation_report(T_c, T_h, c_val, h_val, mu, sd, standardize, normalize_eps)

    return {
        "T_c": T_c, "T_h": T_h, "mu": mu, "sd": sd,
        "standardize": standardize, "affine": affine, "normalize_eps": normalize_eps,
        "alpha": alpha, "rho_c": rho_c, "rho_h": rho_h, **report,
    }


# ----------------------------------------------------------------------

def save_operators(path: str, ops: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        T_c=ops["T_c"], T_h=ops["T_h"], mu=ops["mu"], sd=ops["sd"],
        standardize=np.array(ops["standardize"]),
        affine=np.array(ops["affine"]),
        normalize_eps=np.array(ops["normalize_eps"]),
    )
    print(f"\nSaved operators -> {path}")


def main():
    p = argparse.ArgumentParser(description="Fit T_c and T_h from labeled trajectories.")
    p.add_argument("--in", dest="inp", type=str, default="results/trajectories.npz")
    p.add_argument("--out", type=str, default="results/operators.npz")
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-affine", action="store_true")
    p.add_argument("--no-normalize-eps", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 68)
    print("STEP 2 — FITTING T_c AND T_h  (whiteboard closed-form ridge LS)")
    print("=" * 68)
    print(f"state dims: {STATE_NAMES}")

    success, crash = load_trajectories(args.inp)
    print(f"loaded {len(success)} success / {len(crash)} crash trajectories")

    ops = fit_operators(
        success, crash,
        alpha=args.alpha,
        standardize=not args.no_standardize,
        affine=not args.no_affine,
        normalize_eps=not args.no_normalize_eps,
        seed=args.seed,
    )
    save_operators(args.out, ops)


if __name__ == "__main__":
    main()
