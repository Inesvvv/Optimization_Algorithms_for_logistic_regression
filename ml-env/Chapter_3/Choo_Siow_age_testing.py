"""
File created 04/04/2026
Run SISTA with Newton method on the Choo-Siow aggregate marriage data
(Choo & Siow, 2006 JPE).

Goal: discover which age-related features drive who marries whom.
Types are 60 age groups (ages 16–75) for each side; basis functions capture
age difference, age gap penalties, assortative matching indicators, etc.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter 2'))
from sista_algorithm import sista, sista_newton

# =============================================================================
# 1. Load Choo-Siow aggregate data (60 age groups per side)
# =============================================================================
choo_siow_path = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/marriage-ChooSiow/'

n_singles = pd.read_csv(choo_siow_path + 'n_singles.txt', sep='\t', header=None).values
marr      = pd.read_csv(choo_siow_path + 'marr.txt', sep='\t', header=None).values
n_avail   = pd.read_csv(choo_siow_path + 'n_avail.txt', sep='\t', header=None).values

ages = np.arange(16, 76)
nTypes = len(ages)

p = n_avail[:, 0].astype(float)
q = n_avail[:, 1].astype(float)
hat_pi = marr.astype(float)

hat_pi += 1e-2

print("=== Choo-Siow aggregate data ===")
print(f"  Types (age groups): {nTypes}  (ages {ages[0]}–{ages[-1]})")
print(f"  hat_pi shape: {hat_pi.shape}")
print(f"  Total marriages: {hat_pi.sum():,.0f}")

# =============================================================================
# 2. Build basis functions D (K x N x N)
#    Each D[k, i, j] measures how feature k interacts between man-type i
#    and woman-type j. SISTA will select which features actually matter.
# =============================================================================
age_m = ages[:, None] * np.ones((1, nTypes))
age_w = np.ones((nTypes, 1)) * ages[None, :]

basis_functions = {
    'age_diff':        age_m - age_w,
    'age_diff_sq':     (age_m - age_w)**2,
    'abs_age_diff':    np.abs(age_m - age_w),
    'age_sum':         age_m + age_w,
    'age_product':     age_m * age_w,
    'same_age':        (age_m == age_w).astype(float),
    'man_older_5plus': (age_m - age_w >= 5).astype(float),
    'close_age_3':     (np.abs(age_m - age_w) <= 3).astype(float),
}

D = np.array(list(basis_functions.values()))
basis_names = list(basis_functions.keys())
K = D.shape[0]

for k in range(K):
    s = D[k].std()
    if s > 0:
        D[k] /= s

print(f"\nBasis functions: K={K}")
for i, name in enumerate(basis_names):
    print(f"  D[{i:2d}] = {name}")

# =============================================================================
# 3. Run SISTA (proximal gradient) vs SISTA (Newton)
# =============================================================================
gamma = 5e-3
n_iters = 2000
sinkhorn_inner = 20

print(f"\n{'=' * 65}")
print(f"Running SISTA:  gamma={gamma}, n_iters={n_iters}, sinkhorn_inner={sinkhorn_inner}")
print(f"{'=' * 65}")

t0 = time.time()
beta_pg, u_pg, v_pg, hist_pg = sista(
    p, q, hat_pi, D,
    gamma=gamma, rho=0.05, n_iters=n_iters,
    sinkhorn_inner=sinkhorn_inner, tol=1e-10
)
time_pg = time.time() - t0

t0 = time.time()
beta_nw, u_nw, v_nw, hist_nw = sista_newton(
    p, q, hat_pi, D,
    gamma=gamma, n_iters=n_iters,
    sinkhorn_inner=sinkhorn_inner, tol=1e-10, mu=1e-5
)
time_nw = time.time() - t0

# =============================================================================
# 4. Results
# =============================================================================
print(f"\n{'=' * 65}")
print("RESULTS: Which age features matter for marriage matching?")
print(f"{'=' * 65}")
print(f"\n{'Feature':<30} {'ProxGrad β':>12} {'Newton β':>12}")
print("-" * 55)
for k in range(K):
    marker = " ***" if abs(beta_nw[k]) > 1e-4 else ""
    print(f"{basis_names[k]:<30} {beta_pg[k]:>12.6f} {beta_nw[k]:>12.6f}{marker}")

print(f"\n{'Metric':<25} {'ProxGrad':>12} {'Newton':>12}")
print("-" * 50)
print(f"{'Iterations':<25} {len(hist_pg):>12} {len(hist_nw):>12}")
print(f"{'Wall time':<25} {time_pg:>11.3f}s {time_nw:>11.3f}s")
print(f"{'Final objective':<25} {hist_pg[-1]:>12.6f} {hist_nw[-1]:>12.6f}")

sp_pg = np.sum(np.abs(beta_pg) < 1e-4)
sp_nw = np.sum(np.abs(beta_nw) < 1e-4)
print(f"{'Zeros (sparsity)':<25} {f'{sp_pg}/{K}':>12} {f'{sp_nw}/{K}':>12}")

selected_pg = [basis_names[k] for k in range(K) if abs(beta_pg[k]) > 1e-4]
selected_nw = [basis_names[k] for k in range(K) if abs(beta_nw[k]) > 1e-4]
print(f"\nSelected features (ProxGrad): {selected_pg}")
print(f"Selected features (Newton):   {selected_nw}")

# =============================================================================
# 5. Visualization
# =============================================================================
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    subplot_titles=["Convergence", "Selected Features (Newton)"],
    horizontal_spacing=0.14,
)

opt_val = min(hist_pg[-1], hist_nw[-1])

for name, hist, color in [
    ("Proximal Gradient", hist_pg, "#FF6B35"),
    ("Newton",            hist_nw, "#4ECDC4"),
]:
    subopt = np.array(hist) - opt_val + 1e-14
    fig.add_trace(go.Scatter(
        x=list(range(len(hist))),
        y=subopt,
        mode="lines",
        name=name,
        line=dict(color=color, width=2.5),
    ), row=1, col=1)

colors_bar = ["#4ECDC4" if abs(b) > 1e-4 else "rgba(100,100,100,0.4)" for b in beta_nw]

fig.add_trace(go.Bar(
    y=basis_names,
    x=beta_nw,
    orientation="h",
    name="Newton β",
    marker_color=colors_bar,
    showlegend=False,
), row=1, col=2)

fig.update_layout(
    title=dict(
        text="SISTA on Choo-Siow: Which age features matter for marriage?",
        font=dict(size=17, color="#F5F5F0", family="Georgia, serif"),
    ),
    template="plotly_dark",
    paper_bgcolor="#0B0C10",
    plot_bgcolor="#0B0C10",
    font=dict(family="Georgia, serif", color="#E8E8E0"),
    height=600,
    width=1200,
    legend=dict(orientation="h", y=-0.08, x=0.0, font=dict(size=12)),
)

fig.update_yaxes(row=1, col=1, title="F(β) − F*", type="log",
                 gridcolor="rgba(50,50,50,0.3)")
fig.update_xaxes(row=1, col=1, title="Iteration",
                 gridcolor="rgba(50,50,50,0.3)")
fig.update_xaxes(row=1, col=2, title="β coefficient",
                 gridcolor="rgba(50,50,50,0.3)", zeroline=True,
                 zerolinecolor="rgba(100,100,100,0.5)")
fig.update_yaxes(row=1, col=2, gridcolor="rgba(50,50,50,0.1)",
                 categoryorder="array", categoryarray=basis_names[::-1])

fig.show()
