"""
Created 15/02/2026
This file visualizes LASSO regularization + compares with Ridge

"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================
# Feature data (same as the JSX version)
# =============================================
FEATURES = [
    {"name": "Income",      "color": "#FF6B35", "true_coef":  0.82},
    {"name": "Age",          "color": "#F7C59F", "true_coef":  0.54},
    {"name": "Credit Score", "color": "#4ECDC4", "true_coef":  0.71},
    {"name": "Debt Ratio",   "color": "#A8DADC", "true_coef": -0.63},
    {"name": "Employment",   "color": "#C9B1FF", "true_coef":  0.28},
    {"name": "Location",     "color": "#FF8FA3", "true_coef":  0.12},
    {"name": "Browser",      "color": "#B5EAD7", "true_coef": -0.07},
    {"name": "Day of Week",  "color": "#FFDAC1", "true_coef":  0.04},
]

def lasso_coef(true_coef, lam):
    """Soft thresholding: sign(β) * max(|β| - λ, 0)"""
    return np.sign(true_coef) * max(abs(true_coef) - lam, 0)

def ridge_coef(true_coef, lam):
    """Ridge shrinkage: β / (1 + 2.2λ)"""
    return true_coef / (1 + lam * 2.2)

# =============================================
# Generate frames for the slider animation
# =============================================
lambda_values = np.round(np.linspace(0, 0.9, 46), 2)
names = [f["name"] for f in FEATURES]
colors = [f["color"] for f in FEATURES]
true_coefs = [f["true_coef"] for f in FEATURES]

# =============================================
# Figure 1: Coefficient comparison (LASSO vs Ridge)
# =============================================
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.6, 0.4],
    subplot_titles=["Coefficient Shrinkage", "Geometric Intuition (L1 vs L2)"],
    horizontal_spacing=0.12,
)

# Initial state (lambda = 0): LASSO bars
lasso_vals_init = [lasso_coef(c, 0) for c in true_coefs]
ridge_vals_init = [ridge_coef(c, 0) for c in true_coefs]

# LASSO bars
fig.add_trace(go.Bar(
    y=names, x=lasso_vals_init,
    orientation="h",
    marker_color=colors,
    name="LASSO",
    text=[f"{v:.3f}" for v in lasso_vals_init],
    textposition="outside",
    textfont=dict(size=11),
), row=1, col=1)

# Ridge bars (semi-transparent, behind)
fig.add_trace(go.Bar(
    y=names, x=ridge_vals_init,
    orientation="h",
    marker_color=["rgba(78,205,196,0.35)"] * len(FEATURES),
    name="Ridge",
    text=[f"{v:.3f}" for v in ridge_vals_init],
    textposition="outside",
    textfont=dict(size=11, color="rgba(78,205,196,0.7)"),
), row=1, col=1)

# =============================================
# Geometric intuition: L1 diamond + L2 circle + OLS contours
# =============================================
# L1 diamond
theta_diamond = np.linspace(0, 2 * np.pi, 200)
r_l1 = 1.0
x_diamond = r_l1 * np.array([np.cos(t) / (abs(np.cos(t)) + abs(np.sin(t))) for t in theta_diamond])
y_diamond = r_l1 * np.array([np.sin(t) / (abs(np.cos(t)) + abs(np.sin(t))) for t in theta_diamond])

fig.add_trace(go.Scatter(
    x=x_diamond, y=y_diamond,
    mode="lines", fill="toself",
    fillcolor="rgba(255,107,53,0.08)",
    line=dict(color="#FF6B35", width=2),
    name="L1 Ball (LASSO)",
), row=1, col=2)

# L2 circle
theta_circle = np.linspace(0, 2 * np.pi, 200)
r_l2 = 0.85
fig.add_trace(go.Scatter(
    x=r_l2 * np.cos(theta_circle),
    y=r_l2 * np.sin(theta_circle),
    mode="lines", fill="toself",
    fillcolor="rgba(78,205,196,0.06)",
    line=dict(color="#4ECDC4", width=2, dash="dash"),
    name="L2 Ball (Ridge)",
), row=1, col=2)

# OLS loss contour ellipses
ols_center = (1.2, 0.8)
for scale in [0.4, 0.7, 1.1]:
    ex = ols_center[0] + scale * 0.9 * np.cos(theta_circle)
    ey = ols_center[1] + scale * 0.5 * np.sin(theta_circle)
    fig.add_trace(go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.25)", width=1),
        showlegend=False,
    ), row=1, col=2)

# OLS solution point
fig.add_trace(go.Scatter(
    x=[ols_center[0]], y=[ols_center[1]],
    mode="markers+text",
    marker=dict(size=8, color="#888"),
    text=["β̂ OLS"], textposition="top right",
    textfont=dict(size=11, color="#888"),
    showlegend=False,
), row=1, col=2)

# LASSO solution point (moves with slider)
fig.add_trace(go.Scatter(
    x=[ols_center[0]], y=[ols_center[1]],
    mode="markers+text",
    marker=dict(size=10, color="#FF6B35", symbol="diamond"),
    text=["LASSO"], textposition="bottom left",
    textfont=dict(size=11, color="#FF6B35"),
    name="LASSO solution",
    showlegend=False,
), row=1, col=2)

# Axes through origin
fig.add_hline(y=0, line=dict(color="rgba(100,100,100,0.3)", width=1), row=1, col=2)
fig.add_vline(x=0, line=dict(color="rgba(100,100,100,0.3)", width=1), row=1, col=2)

# =============================================
# Build slider frames
# =============================================
frames = []
for lam in lambda_values:
    lasso_vals = [lasso_coef(c, lam) for c in true_coefs]
    ridge_vals = [ridge_coef(c, lam) for c in true_coefs]
    zeroed = sum(1 for v in lasso_vals if v == 0)

    # Color: dim eliminated features
    bar_colors = [
        "rgba(60,60,60,0.4)" if lasso_vals[i] == 0 else colors[i]
        for i in range(len(FEATURES))
    ]
    bar_text = [
        f"{v:.3f}  ✕" if v == 0 else f"{v:.3f}"
        for v in lasso_vals
    ]

    # Move LASSO solution toward a corner of the diamond
    t = lam / 0.9
    lasso_x = ols_center[0] * (1 - t) + 1.0 * t  # move toward (1, 0) corner
    lasso_y = ols_center[1] * (1 - t * 1.3)
    lasso_y = max(lasso_y, 0)

    frames.append(go.Frame(
        data=[
            go.Bar(y=names, x=lasso_vals, orientation="h",
                   marker_color=bar_colors, name="LASSO",
                   text=bar_text, textposition="outside",
                   textfont=dict(size=11)),
            go.Bar(y=names, x=ridge_vals, orientation="h",
                   marker_color=["rgba(78,205,196,0.35)"] * len(FEATURES),
                   name="Ridge",
                   text=[f"{v:.3f}" for v in ridge_vals],
                   textposition="outside",
                   textfont=dict(size=11, color="rgba(78,205,196,0.7)")),
            # Keep geometry traces unchanged (indices 2-7)
            go.Scatter(x=x_diamond, y=y_diamond),  # L1
            go.Scatter(x=r_l2 * np.cos(theta_circle), y=r_l2 * np.sin(theta_circle)),  # L2
            *[go.Scatter() for _ in range(3)],  # ellipses
            go.Scatter(x=[ols_center[0]], y=[ols_center[1]]),  # OLS point
            go.Scatter(  # LASSO point moves
                x=[lasso_x], y=[lasso_y],
                mode="markers+text",
                marker=dict(size=10, color="#FF6B35", symbol="diamond"),
                text=[f"LASSO (λ={lam:.2f})"], textposition="bottom left",
                textfont=dict(size=11, color="#FF6B35"),
            ),
        ],
        name=str(lam),
        layout=go.Layout(
            title_text=f"LASSO Regularization — λ = {lam:.2f}   |   "
                       f"Features eliminated: {zeroed}/{len(FEATURES)}"
        ),
    ))

fig.frames = frames

# =============================================
# Slider + Play button
# =============================================
sliders = [dict(
    active=0,
    currentvalue={"prefix": "λ = ", "font": {"size": 16, "color": "#FF6B35"}},
    pad={"t": 40},
    steps=[dict(
        method="animate",
        args=[[str(lam)], dict(mode="immediate", frame=dict(duration=50, redraw=True),
                                transition=dict(duration=30))],
        label=f"{lam:.2f}",
    ) for lam in lambda_values],
)]

updatemenus = [dict(
    type="buttons",
    showactive=False,
    y=-0.08, x=0.08,
    buttons=[
        dict(label="▶ Animate",
             method="animate",
             args=[None, dict(frame=dict(duration=80, redraw=True),
                              transition=dict(duration=50),
                              fromcurrent=True, mode="immediate")]),
        dict(label="⏸ Pause",
             method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate")]),
    ],
)]

# =============================================
# Layout
# =============================================
fig.update_layout(
    title=dict(
        text="LASSO Regularization — λ = 0.00   |   Features eliminated: 0/8",
        font=dict(size=18, color="#F5F5F0", family="Georgia, serif"),
    ),
    template="plotly_dark",
    paper_bgcolor="#0B0C10",
    plot_bgcolor="#0B0C10",
    font=dict(family="Georgia, serif", color="#E8E8E0"),
    barmode="overlay",
    sliders=sliders,
    updatemenus=updatemenus,
    height=620,
    width=1100,
    legend=dict(
        orientation="h", y=-0.15, x=0.0,
        font=dict(size=12),
    ),
    # Annotations for key insight
    annotations=[
        dict(
            text="<b>LASSO</b> (L1): shrinks to <b>exact zero</b> → feature selection<br>"
                 "<b>Ridge</b> (L2): shrinks toward zero but <b>never reaches it</b><br><br>"
                 "Soft threshold: β̂ = sign(β)·max(|β| − λ, 0)",
            xref="paper", yref="paper",
            x=0.99, y=0.02,
            showarrow=False,
            font=dict(size=11, color="#888"),
            align="right",
            bgcolor="rgba(15,15,15,0.8)",
            bordercolor="#1a1a1a",
            borderwidth=1,
            borderpad=10,
        ),
    ],
)

# Left plot (bars) axis
fig.update_xaxes(range=[-0.85, 1.0], title="Coefficient value", row=1, col=1,
                 gridcolor="rgba(50,50,50,0.3)", zeroline=True,
                 zerolinecolor="rgba(100,100,100,0.5)", zerolinewidth=1)
fig.update_yaxes(row=1, col=1, gridcolor="rgba(50,50,50,0.1)")

# Right plot (geometry) axis
fig.update_xaxes(range=[-2, 2.5], title="β₁", row=1, col=2,
                 gridcolor="rgba(50,50,50,0.15)", scaleanchor="y2", scaleratio=1)
fig.update_yaxes(range=[-2, 2.5], title="β₂", row=1, col=2,
                 gridcolor="rgba(50,50,50,0.15)")

fig.show()
