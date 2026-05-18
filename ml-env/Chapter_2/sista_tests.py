"""
SISTA Comparison Tests: Proximal Gradient vs Newton
====================================================

Design principle: start from a single BASELINE configuration, then change
exactly ONE parameter per test so that every difference in results can be
attributed to that single change.

BASELINE (shared by all low-dim tests):
    K=8, N=15, gamma=1e-3, rho=0.1, sinkhorn_inner=10, n_iters=20000, tol=1e-12

Test 1 — Baseline:        run both methods to convergence, no cap that matters
Test 2 — Larger step:     rho=0.5              (everything else = baseline)
Test 3 — Few Sinkhorn:    sinkhorn_inner=1     (everything else = baseline)
Test 4 — High-dim:        K=200                (+ necessary adjustments explained below)
Test 5 — High-dim + few:  K=200, sinkhorn_inner=1

For the high-dimensional tests (4 & 5), three parameters must change alongside K:
  - gamma: reduced to 1e-4 because with 200 basis functions and only 3 non-zero,
    the baseline gamma=1e-3 over-penalizes and drives everything to zero.
  - rho: reduced to 0.01 because the Lipschitz constant of the gradient grows
    with K, and the step size must stay below its reciprocal for descent.
  - N: reduced to 10 to keep the Sinkhorn sub-problem cheap so that timing
    reflects the effect of large K, not of a large transport problem.
These are not free choices — they are forced by the change in K.
"""

import numpy as np
import time
import sys

# ─────────────────────────────────────────────────
# Import algorithm functions from your existing file
# ─────────────────────────────────────────────────
from sista_algorithm import (
    logsumexp, soft_threshold, build_cost,
    sinkhorn_one_pass, compute_pi, sista_objective,
    sista, sista_newton,
)


# ─────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────
def generate_problem(K, N, n_nonzero=3, seed=42):
    """
    Generate a synthetic matching problem with a known sparse ground truth.
    Returns marginals (p, q), observed matching hat_pi, basis matrices D,
    and true_beta.
    """
    np.random.seed(seed)

    # Symmetric basis matrices
    D = np.random.randn(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    # Sparse ground truth: place non-zero coefficients at fixed positions
    true_beta = np.zeros(K)
    positions = [1, min(4, K - 1), min(6, K - 1)]
    values = [1.2, -0.8, 0.5]
    for i in range(min(n_nonzero, K)):
        true_beta[positions[i]] = values[i]

    n_true_zeros = K - np.sum(np.abs(true_beta) > 0)

    # Build observed matching from ground truth
    C_true = build_cost(true_beta, D)
    hat_pi = np.exp(-C_true)
    hat_pi = hat_pi / hat_pi.sum()
    p = hat_pi.sum(axis=1)
    q = hat_pi.sum(axis=0)

    return p, q, hat_pi, D, true_beta, n_true_zeros


def generate_highdim_problem(K, N, seed=42):
    """
    Generate a high-dimensional problem (K >> 8) with 3 non-zero coefficients
    spread across the parameter vector.
    """
    np.random.seed(seed)

    D = np.random.randn(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    true_beta = np.zeros(K)
    true_beta[10] = 0.8
    true_beta[50] = -0.6
    true_beta[99] = 0.3
    n_true_zeros = K - 3

    C_true = build_cost(true_beta, D)
    hat_pi = np.exp(-C_true)
    hat_pi = hat_pi / hat_pi.sum()
    p = hat_pi.sum(axis=1)
    q = hat_pi.sum(axis=0)

    return p, q, hat_pi, D, true_beta, n_true_zeros


# ─────────────────────────────────────────────────
# Run and report
# ─────────────────────────────────────────────────
def run_and_report(label, description, p, q, hat_pi, D, true_beta, n_true_zeros,
                   gamma, rho, sinkhorn_inner, n_iters, changed_param=""):
    K = D.shape[0]

    # --- ProxGrad ---
    t0 = time.time()
    b_pg, _, _, h_pg = sista(
        p, q, hat_pi, D,
        gamma=gamma, rho=rho, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=1e-12,
    )
    time_pg = time.time() - t0

    # --- Newton ---
    t0 = time.time()
    b_nw, _, _, h_nw = sista_newton(
        p, q, hat_pi, D,
        gamma=gamma, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=1e-12,
    )
    time_nw = time.time() - t0

    # Sparsity
    sp_pg = np.sum(np.abs(b_pg) < 1e-4)
    sp_nw = np.sum(np.abs(b_nw) < 1e-4)

    # Print
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {description}")
    if changed_param:
        print(f"  Changed from baseline: {changed_param}")
    print(f"  K={K}, N={p.shape[0]}, gamma={gamma}, rho={rho}, "
          f"sinkhorn_inner={sinkhorn_inner}, n_iters={n_iters}")
    print(f"\n  {'Metric':<22} {'ProxGrad':>14} {'Newton':>14}")
    print(f"  {'-' * 50}")
    print(f"  {'Iterations':<22} {len(h_pg):>14} {len(h_nw):>14}")
    print(f"  {'Wall time':<22} {time_pg:>13.4f}s {time_nw:>13.4f}s")
    print(f"  {'Final objective':<22} {h_pg[-1]:>14.6f} {h_nw[-1]:>14.6f}")
    print(f"  {'Zeros found':<22} {f'{sp_pg}/{K}':>14} {f'{sp_nw}/{K}':>14}"
          f"  (true = {n_true_zeros}/{K})")
    print(f"  {'||beta - beta_true||':<22} {np.linalg.norm(b_pg - true_beta):>14.6f}"
          f" {np.linalg.norm(b_nw - true_beta):>14.6f}")

    # Parameter-by-parameter (only for small K)
    if K <= 20:
        print(f"\n  {'':>6} {'True':>8} {'ProxGrad':>10} {'Newton':>10}")
        print(f"  {'-' * 40}")
        for k in range(K):
            print(f"  beta_{k:<3} {true_beta[k]:>8.4f} {b_pg[k]:>10.4f} {b_nw[k]:>10.4f}")

    return h_pg, h_nw


# ─────────────────────────────────────────────────
# BASELINE parameters (low-dimensional)
# ─────────────────────────────────────────────────
BASELINE = dict(
    K=8, N=15, gamma=1e-3, rho=0.1,
    sinkhorn_inner=10, n_iters=20_000,
)


if __name__ == "__main__":

    # ==================================================
    # TEST 1 — BASELINE
    # No changes. Both methods run to convergence.
    # This establishes the reference point.
    # ==================================================
    p, q, hat_pi, D, true_beta, n_zeros = generate_problem(
        K=BASELINE["K"], N=BASELINE["N"],
    )
    h1_pg, h1_nw = run_and_report(
        label="TEST 1: Baseline",
        description="Standard low-dimensional problem, both methods run to convergence.",
        p=p, q=q, hat_pi=hat_pi, D=D,
        true_beta=true_beta, n_true_zeros=n_zeros,
        gamma=BASELINE["gamma"],
        rho=BASELINE["rho"],
        sinkhorn_inner=BASELINE["sinkhorn_inner"],
        n_iters=BASELINE["n_iters"],
    )

    # ==================================================
    # TEST 2 — LARGER STEP SIZE
    # Only change: rho = 0.5 (baseline = 0.1)
    # Purpose: test whether Newton's advantage in Test 1
    # was due to curvature info or just ProxGrad under-stepping.
    # ==================================================
    h2_pg, h2_nw = run_and_report(
        label="TEST 2: Larger step size",
        description="Same problem as Test 1, but with rho=0.5 instead of 0.1.",
        p=p, q=q, hat_pi=hat_pi, D=D,
        true_beta=true_beta, n_true_zeros=n_zeros,
        gamma=BASELINE["gamma"],
        rho=0.5,
        sinkhorn_inner=BASELINE["sinkhorn_inner"],
        n_iters=BASELINE["n_iters"],
        changed_param="rho: 0.1 -> 0.5",
    )

    # ==================================================
    # TEST 3 — FEW SINKHORN ITERATIONS
    # Only change: sinkhorn_inner = 1 (baseline = 10)
    # Purpose: test whether an imprecise transport plan
    # degrades the Hessian approximation that Newton relies on.
    # ==================================================
    h3_pg, h3_nw = run_and_report(
        label="TEST 3: Few Sinkhorn iterations",
        description="Same problem as Test 1, but with only 1 Sinkhorn iteration per outer step.",
        p=p, q=q, hat_pi=hat_pi, D=D,
        true_beta=true_beta, n_true_zeros=n_zeros,
        gamma=BASELINE["gamma"],
        rho=BASELINE["rho"],
        sinkhorn_inner=1,
        n_iters=BASELINE["n_iters"],
        changed_param="sinkhorn_inner: 10 -> 1",
    )

    # ==================================================
    # TEST 4 — HIGH-DIMENSIONAL (K=200)
    # Central change: K = 200 (baseline = 8)
    # Necessary adjustments (forced by large K):
    #   gamma: 1e-3 -> 1e-4  (avoid over-penalizing 200 coefficients)
    #   rho:   0.1  -> 0.01  (Lipschitz constant grows with K)
    #   N:     15   -> 10    (keep Sinkhorn cheap to isolate K effect)
    # sinkhorn_inner stays at 10 so the transport plan is well-resolved.
    # ==================================================
    p4, q4, hat_pi4, D4, true_beta4, n_zeros4 = generate_highdim_problem(
        K=200, N=10,
    )
    h4_pg, h4_nw = run_and_report(
        label="TEST 4: High-dimensional (K=200)",
        description="Large parameter space with well-resolved transport plan.",
        p=p4, q=q4, hat_pi=hat_pi4, D=D4,
        true_beta=true_beta4, n_true_zeros=n_zeros4,
        gamma=1e-4,
        rho=0.01,
        sinkhorn_inner=10,
        n_iters=5_000,
        changed_param="K: 8 -> 200 (gamma, rho, N adjusted as needed)",
    )

    # ==================================================
    # TEST 5 — HIGH-DIMENSIONAL + FEW SINKHORN
    # Same as Test 4, but sinkhorn_inner = 1.
    # Only change from Test 4: sinkhorn_inner: 10 -> 1.
    # Purpose: isolate whether imprecise transport plans
    # compound the instability of Newton in high dimensions.
    # ==================================================
    h5_pg, h5_nw = run_and_report(
        label="TEST 5: High-dimensional + few Sinkhorn iterations",
        description="Same as Test 4, but with only 1 Sinkhorn iteration per outer step.",
        p=p4, q=q4, hat_pi=hat_pi4, D=D4,
        true_beta=true_beta4, n_true_zeros=n_zeros4,
        gamma=1e-4,
        rho=0.01,
        sinkhorn_inner=1,
        n_iters=5_000,
        changed_param="sinkhorn_inner: 10 -> 1 (relative to Test 4)",
    )

    # ==================================================
    # Summary table
    # ==================================================
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Test':<40} {'PG iters':>10} {'NW iters':>10} {'PG obj':>12} {'NW obj':>12}")
    print(f"  {'-' * 64}")
    for name, h_pg, h_nw in [
        ("1: Baseline (K=8)", h1_pg, h1_nw),
        ("2: Larger step (rho=0.5)", h2_pg, h2_nw),
        ("3: Few Sinkhorn (inner=1)", h3_pg, h3_nw),
        ("4: High-dim (K=200)", h4_pg, h4_nw),
        ("5: High-dim + few Sinkhorn", h5_pg, h5_nw),
    ]:
        print(f"  {name:<40} {len(h_pg):>10} {len(h_nw):>10}"
              f" {h_pg[-1]:>12.6f} {h_nw[-1]:>12.6f}")