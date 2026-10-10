import os
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd, pinv
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def get_lifted_data(trajectories, Phi):
    """
    Applies the exact lifting from lifting_function.m:
    y = Phi.T * x
    z = [y; all quadratic combinations y_i * y_j for j <= i]
    Returns Z (inputs) and Y_plus (targets).
    """
    Z_list = []
    Y_plus_list = []
    
    for X in trajectories:
        # X is (8, K)
        Y = Phi.T @ X
        M, K = Y.shape
        
        # Build quadratics
        quads = []
        for i in range(M):
            for j in range(i + 1):
                quads.append(Y[i, :] * Y[j, :])
        
        Z_traj = np.vstack([Y, np.array(quads)]) # (44, K)
        
        # Shift for DMD: z_k -> y_{k+1}
        Z_list.append(Z_traj[:, :-1])
        Y_plus_list.append(Y[:, 1:])
        
    return np.hstack(Z_list), np.hstack(Y_plus_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/ds_data", help="Path to sample_k.mat files")
    parser.add_argument("--train", type=int, default=500, help="Number of training pairs")
    parser.add_argument("--test-start", type=int, default=501, help="Start index for testing")
    parser.add_argument("--test-end", type=int, default=5000, help="End index for testing")
    parser.add_argument("--normalize", action="store_true", help="Divide residuals by sqrt(K-1)")
    args = parser.parse_args()

    print(f"Loading {args.train} training pairs from {args.data}...")
    train_c, train_w = [], []
    for k in range(1, args.train + 1):
        m = loadmat(os.path.join(args.data, f"sample_{k}.mat"))
        train_c.append(m['correct_embeddings'])
        train_w.append(m['wrong_embeddings'])

    # 1. SVD Basis (compute_svd_basis.m)
    print("Computing SVD basis...")
    X_all = np.hstack(train_c + train_w)
    U, _, _ = svd(X_all, full_matrices=False)
    Phi = U[:, :8]  # N_OBS_MODES = 8

    # 2. Lift and assemble matrices (load_training_data.m)
    print("Lifting training data (quadratic features)...")
    Z_c, Y_c_plus = get_lifted_data(train_c, Phi)
    Z_w, Y_w_plus = get_lifted_data(train_w, Phi)
    
    print(f"  Correct transitions: {Z_c.shape[1]}, Lifted dim: {Z_c.shape[0]}")
    print(f"  Wrong transitions:   {Z_w.shape[1]}")

    # 3. Fit Dynamical Systems (fit_dynamical_systems.m)
    print("Fitting Koopman operators (pseudoinverse)...")
    # We only keep the first 8 rows of A to predict the observable Y, exactly like the MATLAB code
    A_c = (Y_c_plus @ pinv(Z_c))
    A_w = (Y_w_plus @ pinv(Z_w))

    # 4. Evaluate (evaluate.m)
    print(f"Evaluating on test samples {args.test_start} to {args.test_end}...")
    delta_E_c = []
    delta_E_w = []
    
    for k in range(args.test_start, args.test_end + 1):
        m = loadmat(os.path.join(args.data, f"sample_{k}.mat"))
        X_c_test = m['correct_embeddings']
        X_w_test = m['wrong_embeddings']
        
        # Lift test samples
        Z_c_test, Y_c_plus_test = get_lifted_data([X_c_test], Phi)
        Z_w_test, Y_w_plus_test = get_lifted_data([X_w_test], Phi)
        
        # Frobenius norms of residuals
        err_c_c = np.linalg.norm(Y_c_plus_test - A_c @ Z_c_test, 'fro')
        err_h_c = np.linalg.norm(Y_c_plus_test - A_w @ Z_c_test, 'fro')
        
        err_c_w = np.linalg.norm(Y_w_plus_test - A_c @ Z_w_test, 'fro')
        err_h_w = np.linalg.norm(Y_w_plus_test - A_w @ Z_w_test, 'fro')
        
        if args.normalize:
            err_c_c /= np.sqrt(Z_c_test.shape[1])
            err_h_c /= np.sqrt(Z_c_test.shape[1])
            err_c_w /= np.sqrt(Z_w_test.shape[1])
            err_h_w /= np.sqrt(Z_w_test.shape[1])
            
        delta_E_c.append(err_h_c - err_c_c) # Should be positive (accurate)
        delta_E_w.append(err_h_w - err_c_w) # Should be negative (hallucinated/wrong)

    delta_E_c = np.array(delta_E_c)
    delta_E_w = np.array(delta_E_w)

    # 5. Threshold sweep and metrics
    print("Sweeping thresholds...")
    # Lander residuals are smaller than LLM residuals, so we sweep a tighter range
    thresholds = np.linspace(-20, 20, 2000) if not args.normalize else np.linspace(-2, 2, 2000)
    
    tpr_list, fpr_list = [], []
    best_f1 = 0
    best_bal_acc = 0
    best_eta = 0
    
    for eta in thresholds:
        # Positive class = hallucination/wrong (delta_E < eta)
        TP = np.sum(delta_E_w < eta)
        FN = np.sum(delta_E_w >= eta)
        FP = np.sum(delta_E_c < eta)
        TN = np.sum(delta_E_c >= eta)
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TPR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        bal_acc = (TPR + (TN / (TN + FP))) / 2 if (TN + FP) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_eta = eta

    roc_auc = auc(fpr_list, tpr_list)
    
    print("\n=== RESULTS ===")
    print(f"AUC:               {roc_auc:.4f}")
    print(f"Best F1:           {best_f1:.4f}")
    print(f"Best Balanced Acc: {best_bal_acc:.4f}")
    print(f"Optimal threshold: {best_eta:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(delta_E_c, bins=50, alpha=0.6, label='Correct (Landing)', color='green')
    plt.hist(delta_E_w, bins=50, alpha=0.6, label='Wrong (Crash)', color='red')
    plt.axvline(best_eta, color='k', linestyle='--', label=f'Threshold $\eta$={best_eta:.2f}')
    plt.xlabel('$\Delta \mathcal{E}$')
    plt.ylabel('Count')
    plt.title('Differential Residual Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr_list, tpr_list, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/paper_ds_port.png')
    print("Saved plot to results/paper_ds_port.png")

if __name__ == "__main__":
    main()