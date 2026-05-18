"""
Damped Newton Tests: Newton vs Damped Newton
=============================================

Design principle: same as sista_tests.py — one shared BASELINE,
change exactly ONE parameter per test.

BASELINE (shared by Tests 1-3, same data):
    K=8, N=15, gamma=1e-3, sinkhorn_inner=10, n_iters=20000, tol=1e-12

Test 1 — Baseline:           standard low-dim problem
Test 2 — Few Sinkhorn:       sinkhorn_inner=1         (only change from Test 1)
Test 3 — Very few Sinkhorn:  sinkhorn_inner=1, noisier basis matrices

HIGH-DIM (shared data for Tests 4-5):
    K=200, N=10, gamma=1e-4, n_iters=5000

Test 4 — High-dim:           sinkhorn_inner=10
Test 5 — High-dim + few:     sinkhorn_inner=1          (only change from Test 4)
"""

import numpy as np
import time

from sista_algorithm import (
    logsumexp, soft_threshold, build_cost,
    sinkhorn_one_pass, compute_pi, sista_objective,
    sista_newton,
)
from damped_newton import sista_damped_newton


# ─────────────────────────────────────────────────
# Data generation (identical to sista_tests.py)
# ─────────────────────────────────────────────────
def generate_problem(K, N, seed=42):
    np.random.seed(seed)
    D = np.random.randn(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    true_beta = np.zeros(K)
    if K <= 8:
        true_beta[1] = 1.2
        true_beta[min(4, K - 1)] = -0.8
        true_beta[min(6, K - 1)] = 0.5
    else:
        true_beta[10] = 0.8
        true_beta[50] = -0.6
        true_beta[99] = 0.3

    n_true_zeros = K - np.sum(np.abs(true_beta) > 0)

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
                   gamma, sinkhorn_inner, n_iters, changed_param=""):
    K = D.shape[0]

    # --- Newton ---
    t0 = time.time()
    b_nw, _, _, h_nw = sista_newton(
        p, q, hat_pi, D,
        gamma=gamma, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=1e-12,
    )
    time_nw = time.time() - t0

    # --- Damped Newton ---
    t0 = time.time()
    b_dnw, _, _, h_dnw, ls_dnw = sista_damped_newton(
        p, q, hat_pi, D,
        gamma=gamma, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=1e-12,
    )
    time_dnw = time.time() - t0

    # Sparsity
    sp_nw = np.sum(np.abs(b_nw) < 1e-4)
    sp_dnw = np.sum(np.abs(b_dnw) < 1e-4)

    # Line search stats
    avg_ls = np.mean(ls_dnw) if ls_dnw else 0
    full_steps = sum(1 for c in ls_dnw if c == 0)

    # Print
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {description}")
    if changed_param:
        print(f"  Changed from baseline: {changed_param}")
    print(f"  K={K}, N={p.shape[0]}, gamma={gamma}, "
          f"sinkhorn_inner={sinkhorn_inner}, n_iters={n_iters}")
    print(f"\n  {'Metric':<25} {'Newton':>14} {'Damped Newton':>14}")
    print(f"  {'-' * 53}")
    print(f"  {'Iterations':<25} {len(h_nw):>14} {len(h_dnw):>14}")
    print(f"  {'Wall time':<25} {time_nw:>13.4f}s {time_dnw:>13.4f}s")
    print(f"  {'Final objective':<25} {h_nw[-1]:>14.6f} {h_dnw[-1]:>14.6f}")
    print(f"  {'Zeros found':<25} {f'{sp_nw}/{K}':>14} {f'{sp_dnw}/{K}':>14}"
          f"  (true = {n_true_zeros}/{K})")
    print(f"  {'||beta - beta_true||':<25} {np.linalg.norm(b_nw - true_beta):>14.6f}"
          f" {np.linalg.norm(b_dnw - true_beta):>14.6f}")
    print(f"  {'Avg backtracks/iter':<25} {'---':>14} {avg_ls:>14.2f}")
    print(f"  {'Full Newton steps':<25} {'---':>14} "
          f"{full_steps}/{len(h_dnw)}")

    if K <= 20:
        print(f"\n  {'':>6} {'True':>8} {'Newton':>10} {'Damped':>10}")
        print(f"  {'-' * 40}")
        for k in range(K):
            print(f"  beta_{k:<3} {true_beta[k]:>8.4f} {b_nw[k]:>10.4f} {b_dnw[k]:>10.4f}")

    return h_nw, h_dnw


if __name__ == "__main__":

    # ==================================================
    # Generate shared low-dim data (used by Tests 1-3)
    # ==================================================
    p, q, hat_pi, D, true_beta, n_zeros = generate_problem(K=8, N=15, seed=42)

    # ==================================================
    # TEST 1 — BASELINE
    # ==================================================
    h1_nw, h1_dnw = run_and_report(
        label="TEST 1: Baseline",
        description="Low-dimensional, well-resolved transport plan.",
        p=p, q=q, hat_pi=hat_pi, D=D,
        true_beta=true_beta, n_true_zeros=n_zeros,
        gamma=1e-3, sinkhorn_inner=10, n_iters=20_000,
    )

    # ==================================================
    # TEST 2 — FEW SINKHORN ITERATIONS
    # Only change: sinkhorn_inner = 1 (baseline = 10)
    # Same data as Test 1.
    # ==================================================
    h2_nw, h2_dnw = run_and_report(
        label="TEST 2: Few Sinkhorn iterations",
        description="Same as Test 1, but sinkhorn_inner=1.",
        p=p, q=q, hat_pi=hat_pi, D=D,
        true_beta=true_beta, n_true_zeros=n_zeros,
        gamma=1e-3, sinkhorn_inner=1, n_iters=20_000,
        changed_param="sinkhorn_inner: 10 -> 1",
    )

    # ==================================================
    # Generate shared high-dim data (used by Tests 3-4)
    # ==================================================
    p4, q4, hat_pi4, D4, true_beta4, n_zeros4 = generate_problem(K=200, N=10, seed=42)

    # ==================================================
    # TEST 3 — HIGH-DIMENSIONAL
    # ==================================================
    h3_nw, h3_dnw = run_and_report(
        label="TEST 3: High-dimensional (K=200)",
        description="Large parameter space, well-resolved transport plan.",
        p=p4, q=q4, hat_pi=hat_pi4, D=D4,
        true_beta=true_beta4, n_true_zeros=n_zeros4,
        gamma=1e-4, sinkhorn_inner=10, n_iters=5_000,
    )

    # ==================================================
    # TEST 4 — HIGH-DIMENSIONAL + FEW SINKHORN
    # Only change from Test 3: sinkhorn_inner = 1
    # ==================================================
    h4_nw, h4_dnw = run_and_report(
        label="TEST 4: High-dimensional + few Sinkhorn",
        description="Same as Test 3, but sinkhorn_inner=1.",
        p=p4, q=q4, hat_pi=hat_pi4, D=D4,
        true_beta=true_beta4, n_true_zeros=n_zeros4,
        gamma=1e-4, sinkhorn_inner=1, n_iters=5_000,
        changed_param="sinkhorn_inner: 10 -> 1 (relative to Test 3)",
    )

    # ==================================================
    # Summary
    # ==================================================
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Test':<40} {'NW iters':>10} {'DNW iters':>10} {'NW obj':>12} {'DNW obj':>12}")
    print(f"  {'-' * 84}")
    for name, h_nw, h_dnw in [
        ("1: Baseline (K=8, inner=10)", h1_nw, h1_dnw),
        ("2: Few Sinkhorn (K=8, inner=1)", h2_nw, h2_dnw),
        ("3: High-dim (K=200, inner=10)", h3_nw, h3_dnw),
        ("4: High-dim + few Sinkhorn", h4_nw, h4_dnw),
    ]:
        print(f"  {name:<40} {len(h_nw):>10} {len(h_dnw):>10}"
              f" {h_nw[-1]:>12.6f} {h_dnw[-1]:>12.6f}")