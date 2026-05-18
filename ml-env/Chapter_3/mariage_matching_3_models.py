"""
File created 18/05/2026
Run SISTA with all three methods on the Personality Traits marriage matching data
(Galichon & Dupuy, 2014 JPE):
  1. Proximal Gradient Descent
  2. Newton
  3. Damped Newton

Goal: discover which personality/physical traits actually matter in who marries whom,
and compare convergence performance across the three optimization variants.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter_2'))
from sista_algorithm import sista, sista_newton
from damped_newton import sista_damped_newton

# =============================================================================
# 1. Load personality traits data
# =============================================================================
trame_path = 'https://raw.githubusercontent.com/TraME-Project/TraME-Datasets/master/'
df = pd.read_csv(trame_path + 'personality_traits/raw/data.csv', sep=';')
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} married couples")

# =============================================================================
# 2. Define types by (education x health)
# =============================================================================
df['type_m'] = df['educm'].astype(int).astype(str) + '_' + df['healthm'].astype(int).astype(str)
df['type_v'] = df['educv'].astype(int).astype(str) + '_' + df['healthv'].astype(int).astype(str)

all_types = sorted(set(df['type_m'].unique()) | set(df['type_v'].unique()))
type_map = {lab: i for i, lab in enumerate(all_types)}

df['tid_m'] = df['type_m'].map(type_map)
df['tid_v'] = df['type_v'].map(type_map)

N = len(all_types)
print(f"Shared type set: {N} types")

# =============================================================================
# 3. Build observed matching matrix
# =============================================================================
hat_pi = np.zeros((N, N))
for _, row in df.iterrows():
    hat_pi[int(row['tid_m']), int(row['tid_v'])] += 1

hat_pi += 1e-2
p = hat_pi.sum(axis=1)
q = hat_pi.sum(axis=0)

print(f"hat_pi: {N}x{N}, non-empty cells: {(hat_pi > 0.1).sum()}/{N*N}")

# =============================================================================
# 4. Build basis functions D (K x N x N)
# =============================================================================
trait_pairs = [
    ('heightm', 'heightv', 'height'),
    ('BMIm',    'BMIv',    'BMI'),
    ('consm',   'consv',   'conscientiousness'),
    ('extram',  'extrav',  'extraversion'),
    ('agreem',  'agreev',  'agreeableness'),
    ('emom',    'emov',    'emot_stability'),
    ('autom',   'autov',   'autonomy'),
    ('riskym',  'riskyv',  'risk_taking'),
]

mean_by_type_m = df.groupby('tid_m')[[fm for fm, _, _ in trait_pairs]].mean().reindex(range(N)).fillna(0)
mean_by_type_v = df.groupby('tid_v')[[fv for _, fv, _ in trait_pairs]].mean().reindex(range(N)).fillna(0)

D_list = []
basis_names = []

for fm, fv, name in trait_pairs:
    xm = mean_by_type_m[fm].values[:, None]
    xv = mean_by_type_v[fv].values[None, :]

    interact = xm * xv
    s = interact.std()
    if s > 0:
        interact /= s
    D_list.append(interact)
    basis_names.append(f'{name}_interact')

    diff_sq = (xm - xv) ** 2
    s = diff_sq.std()
    if s > 0:
        diff_sq /= s
    D_list.append(diff_sq)
    basis_names.append(f'{name}_diff_sq')

D = np.array(D_list)
K = D.shape[0]

print(f"Basis functions: K={K}")
for i, name in enumerate(basis_names):
    print(f"  D[{i:2d}] = {name}")

# =============================================================================
# 5. Run all three methods
# =============================================================================
gamma = 5e-3
n_iters = 100_000
sinkhorn_inner = 20

print(f"\n{'=' * 75}")
print(f"  SISTA: gamma={gamma}, n_iters={n_iters}, sinkhorn_inner={sinkhorn_inner}")
print(f"{'=' * 75}")

# --- Proximal Gradient ---
print("\nRunning Proximal Gradient...")
t0 = time.time()
beta_pg, u_pg, v_pg, hist_pg = sista(
    p, q, hat_pi, D,
    gamma=gamma, rho=0.05, n_iters=n_iters,
    sinkhorn_inner=sinkhorn_inner, tol=1e-10
)
time_pg = time.time() - t0
print(f"  Done: {len(hist_pg)} iterations, {time_pg:.3f}s")

# --- Newton ---
print("\nRunning Newton...")
t0 = time.time()
beta_nw, u_nw, v_nw, hist_nw = sista_newton(
    p, q, hat_pi, D,
    gamma=gamma, n_iters=n_iters,
    sinkhorn_inner=sinkhorn_inner, tol=1e-10, mu=1e-5
)
time_nw = time.time() - t0
print(f"  Done: {len(hist_nw)} iterations, {time_nw:.3f}s")

# --- Damped Newton ---
print("\nRunning Damped Newton...")
t0 = time.time()
beta_dnw, u_dnw, v_dnw, hist_dnw, ls_dnw = sista_damped_newton(
    p, q, hat_pi, D,
    gamma=gamma, n_iters=n_iters,
    sinkhorn_inner=sinkhorn_inner, tol=1e-10, mu=1e-5
)
time_dnw = time.time() - t0
avg_ls = np.mean(ls_dnw) if ls_dnw else 0
full_steps = sum(1 for c in ls_dnw if c == 0)
print(f"  Done: {len(hist_dnw)} iterations, {time_dnw:.3f}s")

# =============================================================================
# 6. Coefficient comparison
# =============================================================================
print(f"\n{'=' * 75}")
print("  ESTIMATED AFFINITY COEFFICIENTS")
print(f"{'=' * 75}")
print(f"\n  {'Feature':<30} {'ProxGrad':>10} {'Newton':>10} {'Damped':>10}")
print(f"  {'-' * 62}")
for k in range(K):
    marker = " ***" if abs(beta_dnw[k]) > 1e-4 else ""
    print(f"  {basis_names[k]:<30} {beta_pg[k]:>10.3f} {beta_nw[k]:>10.3f} {beta_dnw[k]:>10.3f}{marker}")

# =============================================================================
# 7. Performance comparison
# =============================================================================
print(f"\n{'=' * 75}")
print("  CONVERGENCE COMPARISON")
print(f"{'=' * 75}")
print(f"\n  {'Metric':<25} {'ProxGrad':>12} {'Newton':>12} {'Damped':>12}")
print(f"  {'-' * 62}")
print(f"  {'Iterations':<25} {len(hist_pg):>12} {len(hist_nw):>12} {len(hist_dnw):>12}")
print(f"  {'Wall time':<25} {time_pg:>11.3f}s {time_nw:>11.3f}s {time_dnw:>11.3f}s")
print(f"  {'Final objective':<25} {hist_pg[-1]:>12.6f} {hist_nw[-1]:>12.6f} {hist_dnw[-1]:>12.6f}")

sp_pg = np.sum(np.abs(beta_pg) < 1e-4)
sp_nw = np.sum(np.abs(beta_nw) < 1e-4)
sp_dnw = np.sum(np.abs(beta_dnw) < 1e-4)
print(f"  {'Sparsity (zeros)':<25} {f'{sp_pg}/{K}':>12} {f'{sp_nw}/{K}':>12} {f'{sp_dnw}/{K}':>12}")

print(f"\n  Damped Newton line search stats:")
print(f"    Avg backtracks/iter: {avg_ls:.2f}")
print(f"    Full Newton steps:   {full_steps}/{len(hist_dnw)}")

# =============================================================================
# 8. Feature selection comparison
# =============================================================================
print(f"\n{'=' * 75}")
print("  FEATURE SELECTION")
print(f"{'=' * 75}")
for beta, label in [
    (beta_pg, "ProxGrad"),
    (beta_nw, "Newton"),
    (beta_dnw, "Damped Newton"),
]:
    selected = [basis_names[k] for k in range(K) if abs(beta[k]) > 1e-4]
    print(f"\n  {label} ({len(selected)} features):")
    for feat in selected:
        idx = basis_names.index(feat)
        print(f"    {feat:<30} beta = {beta[idx]:>8.4f}")

# =============================================================================
# 9. Agreement check: do Newton and Damped Newton select the same features?
# =============================================================================
sel_nw = set(k for k in range(K) if abs(beta_nw[k]) > 1e-4)
sel_dnw = set(k for k in range(K) if abs(beta_dnw[k]) > 1e-4)
sel_pg = set(k for k in range(K) if abs(beta_pg[k]) > 1e-4)

print(f"\n{'=' * 75}")
print("  AGREEMENT")
print(f"{'=' * 75}")
print(f"  Newton vs Damped Newton: {'AGREE' if sel_nw == sel_dnw else 'DISAGREE'}")
print(f"    Newton selects:        {sorted(sel_nw)}")
print(f"    Damped Newton selects: {sorted(sel_dnw)}")
if sel_nw != sel_dnw:
    print(f"    Only in Newton:        {sorted(sel_nw - sel_dnw)}")
    print(f"    Only in Damped:        {sorted(sel_dnw - sel_nw)}")

print(f"\n  ProxGrad vs Newton: {'AGREE' if sel_pg == sel_nw else 'DISAGREE'}")
print(f"    ProxGrad selects:      {sorted(sel_pg)}")
if sel_pg != sel_nw:
    print(f"    Only in ProxGrad:      {sorted(sel_pg - sel_nw)}")
    print(f"    Only in Newton:        {sorted(sel_nw - sel_pg)}")

# =============================================================================
# 10. Print tikz coordinates for convergence plot
# =============================================================================
opt_val = min(hist_pg[-1], hist_nw[-1], hist_dnw[-1])

print(f"\n% --- TikZ coordinates for convergence plot ---")
print(f"% Optimal value: {opt_val:.10f}")

for name, hist, max_pts in [
    ("ProxGrad", hist_pg, 50),
    ("Newton", hist_nw, 50),
    ("Damped Newton", hist_dnw, 50),
]:
    # Subsample for tikz (too many points otherwise)
    n = len(hist)
    if n > max_pts:
        indices = np.linspace(0, n - 1, max_pts, dtype=int)
    else:
        indices = range(n)

    coords = " ".join(
        f"({i}, {max(hist[i] - opt_val, 1e-15):.2e})"
        for i in indices
    )
    print(f"\n% {name} ({n} iterations)")
    print(f"\\addplot coordinates {{ {coords} }};")