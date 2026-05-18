"""
File created 19/03/2026
Run SISTA with Newton method on the Personality Traits marriage matching data
(Galichon & Dupuy, 2014 JPE).

Goal: discover which personality/physical traits actually matter in who marries whom.
Types are defined by (education x health) bins; basis functions capture interactions
of height, BMI, and Big Five personality traits between partners.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Chapter_2'))
from sista_algorithm import sista, sista_newton

# =============================================================================
# 1. Load personality traits data
# =============================================================================
trame_path = 'https://raw.githubusercontent.com/TraME-Project/TraME-Datasets/master/'
df = pd.read_csv(trame_path + 'personality_traits/raw/data.csv', sep=';')
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} married couples")
print(f"Columns: {list(df.columns)}\n")

# =============================================================================
# 2. Define types by (education x health) — shared type set for both sides
#    SISTA expects a square N x N matching matrix, so we use the union of all
#    (education, health) pairs observed across both men and women.
# =============================================================================
df['type_m'] = df['educm'].astype(int).astype(str) + '_' + df['healthm'].astype(int).astype(str)
df['type_v'] = df['educv'].astype(int).astype(str) + '_' + df['healthv'].astype(int).astype(str)

all_types = sorted(set(df['type_m'].unique()) | set(df['type_v'].unique()))
type_map = {lab: i for i, lab in enumerate(all_types)}

df['tid_m'] = df['type_m'].map(type_map)
df['tid_v'] = df['type_v'].map(type_map)

N = len(all_types)
print(f"Shared type set ({N} types): {all_types}")

# =============================================================================
# 3. Build observed matching matrix hat_pi and marginals p, q
#    Add small smoothing (1e-2) to avoid log(0) in Sinkhorn
# =============================================================================
hat_pi = np.zeros((N, N))
for _, row in df.iterrows():
    hat_pi[int(row['tid_m']), int(row['tid_v'])] += 1

hat_pi += 1e-2

p = hat_pi.sum(axis=1)
q = hat_pi.sum(axis=0)

print(f"\nhat_pi shape: {hat_pi.shape}  ({N} x {N} = {N * N} cells)")
print(f"Total couples: {hat_pi.sum():.0f}")
print(f"Non-empty cells: {(hat_pi > 0.1).sum()} / {N * N}")

# =============================================================================
# 4. Build basis functions D (K x nI x nJ)
#    For each continuous trait, compute the type-level mean, then create:
#      - interaction: x_m[i] * x_v[j]   (do similar types match?)
#      - diff_sq:     (x_m[i] - x_v[j])^2  (is mismatch penalized?)
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

# Compute mean feature per type, reindex to the full shared type set
mean_by_type_m = df.groupby('tid_m')[[fm for fm, _, _ in trait_pairs]].mean().reindex(range(N)).fillna(0)
mean_by_type_v = df.groupby('tid_v')[[fv for _, fv, _ in trait_pairs]].mean().reindex(range(N)).fillna(0)

D_list = []
basis_names = []

for fm, fv, name in trait_pairs:
    xm = mean_by_type_m[fm].values[:, None]  # (nI, 1)
    xv = mean_by_type_v[fv].values[None, :]  # (1, nJ)

    # Interaction term
    interact = xm * xv
    s = interact.std()
    if s > 0:
        interact /= s
    D_list.append(interact)
    basis_names.append(f'{name}_interact')

    # Squared difference term
    diff_sq = (xm - xv) ** 2
    s = diff_sq.std()
    if s > 0:
        diff_sq /= s
    D_list.append(diff_sq)
    basis_names.append(f'{name}_diff_sq')

D = np.array(D_list)
K = D.shape[0]

print(f"\nBasis functions: K={K}")
for i, name in enumerate(basis_names):
    print(f"  D[{i:2d}] = {name}")

# =============================================================================
# 5. Run SISTA (proximal gradient) vs SISTA (Newton)
# =============================================================================
gamma = 5e-3
n_iters = 100000
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
# 6. Results
# =============================================================================
print(f"\n{'=' * 65}")
print("RESULTS: Which personality traits matter for marriage matching?")
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
# 7. Visualization
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
        text="SISTA on Personality Traits: Which traits matter for marriage?",
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
