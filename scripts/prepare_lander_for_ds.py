"""
prepare_lander_for_ds.py
========================
Converts the LunarLander trajectory datasets into the exact file format that
the DS repo (hallucination-detection-DS) expects.

The repo's load_training_data.m / evaluate.m read:

    DATA_DIR/sample_1.mat, sample_2.mat, ...

and each file must contain TWO variables:

    correct_embeddings   EMB_DIM x L_c     (one factual / correct trajectory)
    wrong_embeddings     EMB_DIM x L_w     (one hallucinated / wrong trajectory)

For the lander, EMB_DIM = 8 and the "embedding" is simply the physical state
    x_k = (x, y, vx, vy, angle, ang_vel, leg1, leg2) in R^8
so our (8, K) matrices already match the required layout exactly.

Usage
-----
    python prepare_lander_for_ds.py
    python prepare_lander_for_ds.py --n-pairs 10000 --out results/ds_data

Then in MATLAB set  DATA_DIR = 'results/ds_data'  and  EMB_DIM = 8.
"""

import argparse
import os
import numpy as np
from scipy.io import savemat


def load_npz(path):
    """Load a trajectory .npz written by collect_for_advisor.py.

    Returns a list of (8, K) float64 arrays.
    """
    d = np.load(path, allow_pickle=True)
    trajs = list(d["trajectories"])
    out = []
    for t in trajs:
        t = np.asarray(t, dtype=np.float64)
        # Defensive: accept (K, 8) and transpose to (8, K).
        if t.ndim != 2:
            continue
        if t.shape[0] != 8 and t.shape[1] == 8:
            t = t.T
        if t.shape[0] != 8:
            continue
        if t.shape[1] < 3:          # need at least 2 transitions
            continue
        out.append(np.ascontiguousarray(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct", default="results/correct_trajectories.npz")
    ap.add_argument("--wrong", default="results/wrong_trajectories.npz")
    ap.add_argument("--out", default="results/ds_data",
                    help="DATA_DIR for the MATLAB code")
    ap.add_argument("--n-pairs", type=int, default=0,
                    help="how many sample_k.mat files to write (0 = as many as possible)")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="shuffle before pairing so train/test splits are unbiased")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading ...")
    correct = load_npz(args.correct)
    wrong = load_npz(args.wrong)
    print(f"  correct trajectories : {len(correct):6d}")
    print(f"  wrong   trajectories : {len(wrong):6d}")

    rng = np.random.default_rng(args.seed)
    if args.shuffle:
        # IMPORTANT: the MATLAB code splits train/test by FILE INDEX
        # (train = samples 1..TRAIN_LIMIT, test = TEST_START..TEST_END).
        # Shuffling here is what makes that split random rather than
        # "whatever order the episodes happened to be collected in".
        rng.shuffle(correct)
        rng.shuffle(wrong)

    n = min(len(correct), len(wrong))
    if args.n_pairs > 0:
        n = min(n, args.n_pairs)
    print(f"  writing {n} paired sample files")

    os.makedirs(args.out, exist_ok=True)

    len_c, len_w = [], []
    for k in range(n):
        c = correct[k]
        w = wrong[k]
        len_c.append(c.shape[1])
        len_w.append(w.shape[1])
        savemat(
            os.path.join(args.out, f"sample_{k + 1}.mat"),
            {"correct_embeddings": c, "wrong_embeddings": w},
            do_compression=False,
            format="5",
        )
        if (k + 1) % 1000 == 0:
            print(f"    {k + 1}/{n}")

    len_c = np.array(len_c)
    len_w = np.array(len_w)

    print("\ndone.")
    print(f"  DATA_DIR            = {os.path.abspath(args.out)}")
    print(f"  EMB_DIM             = 8")
    print(f"  files written       = {n}")
    print(f"  correct length K    : mean {len_c.mean():7.1f}  min {len_c.min()}  max {len_c.max()}")
    print(f"  wrong   length K    : mean {len_w.mean():7.1f}  min {len_w.min()}  max {len_w.max()}")
    print()
    print("  suggested MATLAB settings:")
    tr = min(500, n // 4)
    print(f"    TRAIN_LIMIT  = {tr}        % ~{tr * int(len_c.mean()):,} transitions")
    print(f"    TEST_START   = {tr + 1}")
    print(f"    TEST_END     = {n}")
    print(f"    tokens_length_threshold = [1 {int(np.percentile(len_w, 25))} "
          f"{int(np.percentile(len_w, 50))} {int(np.percentile(len_w, 75))}]")


if __name__ == "__main__":
    main()
