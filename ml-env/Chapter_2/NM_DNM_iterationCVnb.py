"""
Scaling experiment: How do Newton and Damped Newton iteration counts
evolve as K (number of basis functions) increases?

We fix N=10, gamma=1e-4, sinkhorn_inner=10, n_iters=10000, tol=1e-10,
and sweep K through a range of values. For each K, we generate a
synthetic problem with 3 non-zero coefficients, run both methods,
and record:
  - iteration count (or n_iters if not converged)
  - final objective
  - whether the method converged
  - parameter error

The output is a CSV and a printout that can be pasted into a tikz plot to add to the latex file .
"""

import numpy as np
import time
import csv

from sista_algorithm import (
    build_cost, sista_newton,
)
from damped_newton import sista_damped_newton


def generate_problem(K, N, seed=42):
    np.random.seed(seed)
    D = np.random.randn(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    true_beta = np.zeros(K)
    if K >= 100:
        true_beta[10] = 0.8
        true_beta[50] = -0.6
        true_beta[99] = 0.3
    elif K >= 50:
        true_beta[10] = 0.8
        true_beta[30] = -0.6
        true_beta[49] = 0.3
    elif K >= 8:
        true_beta[1] = 1.2
        true_beta[min(4, K - 1)] = -0.8
        true_beta[min(6, K - 1)] = 0.5

    C_true = build_cost(true_beta, D)
    hat_pi = np.exp(-C_true)
    hat_pi = hat_pi / hat_pi.sum()
    p = hat_pi.sum(axis=1)
    q = hat_pi.sum(axis=0)

    return p, q, hat_pi, D, true_beta


# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────
N = 10
gamma = 1e-4
sinkhorn_inner = 10
n_iters = 10_000
tol = 1e-10
seed = 42

K_values = [8, 16, 32, 50, 75, 100, 150, 200, 300, 400, 500]

results = []

print(f"{'K':>5} | {'NW iters':>10} {'NW obj':>12} {'NW err':>12} {'NW conv':>8} | "
      f"{'DNW iters':>10} {'DNW obj':>12} {'DNW err':>12} {'DNW conv':>8} | "
      f"{'NW time':>8} {'DNW time':>8}")
print("-" * 130)

for K in K_values:
    p, q, hat_pi, D, true_beta = generate_problem(K, N, seed=seed)

    # --- Newton ---
    t0 = time.time()
    b_nw, _, _, h_nw = sista_newton(
        p, q, hat_pi, D,
        gamma=gamma, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=tol,
    )
    time_nw = time.time() - t0
    nw_converged = len(h_nw) < n_iters
    nw_err = np.linalg.norm(b_nw - true_beta)

    # --- Damped Newton ---
    t0 = time.time()
    b_dnw, _, _, h_dnw, ls_dnw = sista_damped_newton(
        p, q, hat_pi, D,
        gamma=gamma, n_iters=n_iters,
        sinkhorn_inner=sinkhorn_inner, tol=tol,
    )
    time_dnw = time.time() - t0
    dnw_converged = len(h_dnw) < n_iters
    dnw_err = np.linalg.norm(b_dnw - true_beta)

    row = {
        "K": K,
        "nw_iters": len(h_nw),
        "nw_obj": h_nw[-1],
        "nw_err": nw_err,
        "nw_converged": nw_converged,
        "nw_time": time_nw,
        "dnw_iters": len(h_dnw),
        "dnw_obj": h_dnw[-1],
        "dnw_err": dnw_err,
        "dnw_converged": dnw_converged,
        "dnw_time": time_dnw,
    }
    results.append(row)

    conv_nw = "yes" if nw_converged else "NO"
    conv_dnw = "yes" if dnw_converged else "NO"
    print(f"{K:>5} | {len(h_nw):>10} {h_nw[-1]:>12.4f} {nw_err:>12.4f} {conv_nw:>8} | "
          f"{len(h_dnw):>10} {h_dnw[-1]:>12.4f} {dnw_err:>12.4f} {conv_dnw:>8} | "
          f"{time_nw:>7.2f}s {time_dnw:>7.2f}s")


# ─────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────
csv_path = "scaling_K_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {csv_path}")


# ─────────────────────────────────────────────────
# Print tikz coordinates for easy copy-paste to latex 
# ─────────────────────────────────────────────────
print("\n% --- TikZ coordinates: Newton iterations vs K ---")
print("% (capped at n_iters if not converged)")
coords_nw = " ".join(f"({r['K']}, {r['nw_iters']})" for r in results)
print(f"\\addplot coordinates {{ {coords_nw} }};")

print("\n% --- TikZ coordinates: Damped Newton iterations vs K ---")
coords_dnw = " ".join(f"({r['K']}, {r['dnw_iters']})" for r in results)
print(f"\\addplot coordinates {{ {coords_dnw} }};")

print("\n% --- TikZ coordinates: Newton final objective vs K ---")
coords_nw_obj = " ".join(f"({r['K']}, {r['nw_obj']:.6f})" for r in results)
print(f"\\addplot coordinates {{ {coords_nw_obj} }};")

print("\n% --- TikZ coordinates: Damped Newton final objective vs K ---")
coords_dnw_obj = " ".join(f"({r['K']}, {r['dnw_obj']:.6f})" for r in results)
print(f"\\addplot coordinates {{ {coords_dnw_obj} }};")

print("\n% --- TikZ coordinates: Newton parameter error vs K ---")
coords_nw_err = " ".join(f"({r['K']}, {r['nw_err']:.4f})" for r in results)
print(f"\\addplot coordinates {{ {coords_nw_err} }};")

print("\n% --- TikZ coordinates: Damped Newton parameter error vs K ---")
coords_dnw_err = " ".join(f"({r['K']}, {r['dnw_err']:.6f})" for r in results)
print(f"\\addplot coordinates {{ {coords_dnw_err} }};")