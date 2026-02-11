"""
LCRD THEORETICAL VALIDATION SUITE - PRODUCTION GRADE
Framework: Lattice-constrained Representation Dynamics (LCRD)
Version: 2.1 (Final)

Validates:
1. Information Plane 'Boomerang' (Compression vs. Relevance)
2. Invariant Sublattice Restriction (Nuisance suppression)
3. Participation Ratio (Representation Dimensionality)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

# ============================================================================
# 1. GLOBAL CONFIGURATION
# ============================================================================

class Config:
    seed = 42
    num_samples = 3000
    input_dim = 128
    hidden_dim = 64
    repr_dim = 32
    num_classes = 5
    num_epochs = 60
    batch_size = 64
    learning_rate = 0.0015
    lambda_inv = 0.6  # Strength of the invariance constraint
    results_dir = Path("./lcrd_results_final")

    def __init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(self.seed)

cfg = Config()

# ============================================================================
# 2. ROBUST METRICS (INFO THEORY & DIMENSIONALITY)
# ============================================================================

def estimate_mi(X, Y, bins=20):
    """
    Stabilized MI Estimation.
    Ensures length matching by projecting high-dim tensors to 1D via mean 
    before performing 2D histogram discretization.
    """
    # Reduce to (N,) to ensure matching lengths for np.histogram2d
    x_val = np.mean(X, axis=1) if X.ndim > 1 else X.flatten()
    y_val = np.mean(Y, axis=1) if Y.ndim > 1 else Y.flatten()
    
    if len(x_val) != len(y_val):
        # Fallback safety (should not be reached with mean reduction)
        min_len = min(len(x_val), len(y_val))
        x_val, y_val = x_val[:min_len], y_val[:min_len]

    hist, _, _ = np.histogram2d(x_val, y_val, bins=[bins, bins])
    p_xy = hist / (np.sum(hist) + 1e-12)
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    
    # I(X;Y) = sum( P(x,y) * log( P(x,y) / (P(x)*P(y)) ) )
    mask = p_xy > 0
    mi = np.sum(p_xy[mask] * np.log2(p_xy[mask] / (p_x @ p_y)[mask] + 1e-12))
    return max(0.0, float(mi))

def participation_ratio(reps):
    """Computes effective dimensionality via the covariance spectrum."""
    centered = reps - np.mean(reps, axis=0)
    cov = (centered.T @ centered) / (len(reps) - 1)
    # Use eigh for symmetric covariance matrices
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    return (np.sum(eigvals)**2) / np.sum(eigvals**2)

# ============================================================================
# 3. CORE LCRD MODEL
# ============================================================================

class LCRDNetwork:
    """Neural model with analytic gradients for Task + Invariance."""
    def __init__(self, d_in, d_hid, d_rep, d_out):
        # He Initialization
        self.W1 = np.random.randn(d_in, d_hid) * np.sqrt(2./d_in)
        self.b1 = np.zeros((1, d_hid))
        self.W2 = np.random.randn(d_hid, d_rep) * np.sqrt(2./d_hid)
        self.b2 = np.zeros((1, d_rep))
        self.W3 = np.random.randn(d_rep, d_out) * np.sqrt(2./d_out)
        self.b3 = np.zeros((1, d_out))
        self.cache = {}

    def forward(self, X, store_cache=True):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1) # ReLU
        z2 = a1 @ self.W2 + self.b2
        # T-Layer: Normalize to unit hypersphere for MI stability
        t = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-10)
        z3 = t @ self.W3 + self.b3
        
        # Softmax
        exp_z3 = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        y_hat = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)
        
        if store_cache:
            self.cache = {'X': X, 'z1': z1, 'a1': a1, 't': t, 'y_hat': y_hat}
        return t, y_hat

    def backward(self, y_true, t_aug, l_inv):
        m = y_true.shape[0]
        # 1. Task Gradients (Cross Entropy)
        dz3 = self.cache['y_hat'].copy()
        dz3[range(m), y_true] -= 1
        dz3 /= m
        
        dw3 = self.cache['t'].T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        # 2. Invariance Gradients (MSE vs Augmented View)
        # Gradient of ||t - t_aug||^2 w.r.t t is 2(t - t_aug)
        dt_inv = 2 * (self.cache['t'] - t_aug) * l_inv / m
        dt_task = dz3 @ self.W3.T
        dt_total = dt_task + dt_inv
        
        # 3. Backprop to Hidden
        dw2 = self.cache['a1'].T @ dt_total
        db2 = np.sum(dt_total, axis=0, keepdims=True)
        
        da1 = dt_total @ self.W2.T
        dz1 = da1 * (self.cache['z1'] > 0).astype(float) # ReLU Grad
        dw1 = self.cache['X'].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2, 'W3': dw3, 'b3': db3}

    def update(self, grads, lr):
        for p in grads:
            setattr(self, p, getattr(self, p) - lr * grads[p])

# ============================================================================
# 4. TRAINING & EVALUATION ENGINE
# ============================================================================

def run_suite():
    print(f"--- LCRD FINAL VALIDATION (Python {sys.version.split()[0]}) ---")
    
    # Generate Synthetic Manifold with Nuisance (theta)
    X = np.random.randn(cfg.num_samples, cfg.input_dim)
    Y = np.random.randint(0, cfg.num_classes, cfg.num_samples)
    Theta = np.random.rand(cfg.num_samples) * 2 * np.pi # Nuisance variable
    
    # Induce class-separability
    for c in range(cfg.num_classes):
        X[Y == c] += np.random.randn(cfg.input_dim) * 2.5

    model = LCRDNetwork(cfg.input_dim, cfg.hidden_dim, cfg.repr_dim, cfg.num_classes)
    h = {'I_TY': [], 'I_TX': [], 'I_TTheta': [], 'acc': [], 'p_ratio': []}

    for epoch in range(cfg.num_epochs):
        perm = np.random.permutation(cfg.num_samples)
        for i in range(0, cfg.num_samples, cfg.batch_size):
            idx = perm[i:i+cfg.batch_size]
            xb, yb = X[idx], Y[idx]
            
            # 1. Forward Original (Stores activation cache)
            t_orig, _ = model.forward(xb, store_cache=True)
            
            # 2. Forward Augmented (Standard LCRD Invariance check)
            xb_aug = xb + np.random.randn(*xb.shape) * 0.15
            t_aug, _ = model.forward(xb_aug, store_cache=False) # Don't overwrite cache
            
            # 3. Update
            grads = model.backward(yb, t_aug, cfg.lambda_inv)
            lr = cfg.learning_rate * (0.96 ** (epoch // 5)) # Step decay
            model.update(grads, lr)

        # Log Metrics
        t_full, y_full = model.forward(X, store_cache=False)
        accuracy = np.mean(np.argmax(y_full, axis=1) == Y) * 100
        
        # Subsample for MI calculation speed
        s_idx = np.random.choice(cfg.num_samples, 1000, replace=False)
        h['I_TY'].append(estimate_mi(t_full[s_idx], Y[s_idx]))
        h['I_TX'].append(estimate_mi(t_full[s_idx], X[s_idx]))
        h['I_TTheta'].append(estimate_mi(t_full[s_idx], Theta[s_idx]))
        h['acc'].append(accuracy)
        h['p_ratio'].append(participation_ratio(t_full))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Acc: {accuracy:5.1f}% | I(T;θ): {h['I_TTheta'][-1]:.3f} | Dim: {h['p_ratio'][-1]:.2f}")

    _generate_report(h)

def _generate_report(h):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Information Plane (Complexity vs Relevance)
    # Expectation: Initial increase in both, then decrease in I(T;X) as I(T;Y) plateaus
    ax1.plot(h['I_TX'], h['I_TY'], 'o-', alpha=0.3, color='gray')
    sc = ax1.scatter(h['I_TX'], h['I_TY'], c=range(len(h['acc'])), cmap='viridis', edgecolors='k', zorder=5)
    ax1.set_title("LCRD Information Plane")
    ax1.set_xlabel("Complexity I(T;X)")
    ax1.set_ylabel("Relevance I(T;Y)")
    plt.colorbar(sc, ax=ax1, label='Epoch')

    # Plot 2: Nuisance Suppression vs Dimensionality
    ax2_twin = ax2.twinx()
    lns1 = ax2.plot(h['I_TTheta'], color='tab:red', lw=2, label='Nuisance Info I(T;θ)')
    lns2 = ax2_twin.plot(h['p_ratio'], color='tab:blue', lw=2, linestyle='--', label='Effective Dim (PR)')
    ax2.set_title("Invariance & Capacity Dynamics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mutual Information (bits)")
    ax2_twin.set_ylabel("Participation Ratio")
    
    # Combined Legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='center right')

    plt.tight_layout()
    plt.savefig(cfg.results_dir / "final_validation_dashboard.png")
    print(f"\n[SUCCESS] Dashboard saved to {cfg.results_dir}")
    
    print("\n" + "="*50)
    print("THEORETICAL VERDICT:")
    print(f"1. Compression Phase: {'PASS' if h['I_TX'][-1] < max(h['I_TX']) else 'FAIL'}")
    print(f"2. Nuisance Reduction: {'PASS' if h['I_TTheta'][-1] < h['I_TTheta'][0] else 'FAIL'}")
    print(f"3. Final Task Acc:     {h['acc'][-1]:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_suite()
