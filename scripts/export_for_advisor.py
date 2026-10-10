"""
scripts/export_for_advisor.py

Turns results/correct_trajectories.npz and results/wrong_trajectories.npz into
plain files the advisor can open directly:

    results/export/correct_csv/traj_00000.csv ... one file per trajectory
    results/export/wrong_csv/traj_00000.csv       (8 rows x K columns, first column = state name)
    results/export/correct_trajectories.mat       MATLAB cell array  trajectories{i} is 8 x K_i
    results/export/wrong_trajectories.mat
    results/export/correct_csv.zip, wrong_csv.zip

Usage:
    python scripts/export_for_advisor.py
"""

import os
import shutil
import numpy as np

RESULTS = "results"
OUT = os.path.join(RESULTS, "export")

DATASETS = {
    "correct": os.path.join(RESULTS, "correct_trajectories.npz"),
    "wrong": os.path.join(RESULTS, "wrong_trajectories.npz"),
}


def export(name: str, path: str) -> None:
    d = np.load(path, allow_pickle=True)
    X = d["trajectories"]
    names = [str(s) for s in d["state_names"]]
    print(f"[{name}] {len(X)} trajectories, first shape {X[0].shape}, "
          f"K range {min(m.shape[1] for m in X)}..{max(m.shape[1] for m in X)}")

    # ---- one CSV per trajectory: 8 rows (states) x K columns (time steps) ----
    csv_dir = os.path.join(OUT, f"{name}_csv")
    os.makedirs(csv_dir, exist_ok=True)
    for i, M in enumerate(X):
        with open(os.path.join(csv_dir, f"traj_{i:05d}.csv"), "w") as f:
            f.write("state," + ",".join(f"k{k+1}" for k in range(M.shape[1])) + "\n")
            for r in range(8):
                f.write(names[r] + "," + ",".join(f"{v:.6f}" for v in M[r]) + "\n")
    shutil.make_archive(csv_dir, "zip", csv_dir)
    print(f"  wrote {len(X)} CSV files -> {csv_dir}.zip")

    # ---- one MATLAB file per dataset ----
    try:
        from scipy.io import savemat
        cell = np.empty((len(X), 1), dtype=object)
        for i, M in enumerate(X):
            cell[i, 0] = M.astype(np.float64)
        savemat(os.path.join(OUT, f"{name}_trajectories.mat"),
                {"trajectories": cell, "lengths": d["lengths"], "state_names": names})
        print(f"  wrote {name}_trajectories.mat")
    except ImportError:
        print("  (scipy not installed -- skipping .mat; pip install scipy)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, path in DATASETS.items():
        if os.path.exists(path):
            export(name, path)
        else:
            print(f"[{name}] MISSING: {path} -- run collect_for_advisor.py first")
    print(f"\nSend the advisor the contents of {OUT}/")
