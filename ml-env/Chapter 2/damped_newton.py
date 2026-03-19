"""
Created on 06/03/2026
This file imports the sista algorithm implemented with the newton method from sista_algorithm.py
and implements the damped newton method applied to sista.

Standard Newton takes the full proximal Newton step each iteration.
Damped Newton adds a backtracking line search: it scales the step by alpha in (0,1]
chosen so the objective actually decreases (Armijo condition).
This makes the method more robust when the quadratic approximation is poor
(far from the optimum, noisy Hessian, ill-conditioned problems).

We compare convergence of both implementations.
"""

import numpy as np
from sista_algorithm import (
    logsumexp, soft_threshold, build_cost,
    sinkhorn_one_pass, compute_pi, sista_objective,
    sista_newton,
)


def sista_damped_newton(p, q, hat_pi, D, beta0=None, u0=None, v0=None,
                        gamma=1e-2, n_iters=500,
                        sinkhorn_inner=1, tol=1e-7, mu=1e-6,
                        alpha_init=1.0, armijo_c=1e-4, armijo_rho=0.5,
                        verbose=False):
    """
    SISTA with Damped Proximal Newton.

    Same as sista_newton but each iteration performs a backtracking line search
    over the step size alpha:
        beta_trial = beta + alpha * (beta_newton - beta)
    accepting the first alpha that satisfies the Armijo sufficient-decrease condition.

    Parameters
    ----------
    alpha_init : float
        Initial step size to try (typically 1.0 = full Newton step).
    armijo_c : float
        Armijo sufficient-decrease parameter (small, e.g. 1e-4).
    armijo_rho : float
        Backtracking shrinkage factor (each rejection multiplies alpha by this).
    """
    N = p.shape[0]
    K = D.shape[0]

    p = p / p.sum()
    q = q / q.sum()
    hat_pi = hat_pi / hat_pi.sum()

    beta = np.zeros(K) if beta0 is None else beta0.astype(float).copy()
    u = np.zeros(N) if u0 is None else u0.astype(float).copy()
    v = np.zeros(N) if v0 is None else v0.astype(float).copy()

    prev_beta = beta.copy()
    history = []
    ls_counts = []

    for t in range(n_iters):
        # (1) cost from beta
        C = build_cost(beta, D)

        # (2) Sinkhorn updates
        for _ in range(sinkhorn_inner):
            u, v = sinkhorn_one_pass(p, q, C, u, v)

        # (3) current plan and objective
        pi = compute_pi(u, v, C)
        f_cur = sista_objective(hat_pi, pi, beta, gamma)
        history.append(f_cur)

        # (4) gradient: g_k = <hat_pi - pi, D[k]>
        diff = hat_pi - pi
        g = np.tensordot(D, diff, axes=([1, 2], [0, 1]))

        # (5a) Hessian diagonal
        D_flat = D.reshape(K, -1)
        pi_flat = pi.ravel()
        mean_k = D_flat @ pi_flat
        mean_sq_k = (D_flat ** 2) @ pi_flat
        H_diag = mean_sq_k - mean_k ** 2 + mu # mu prevents division by zero 

        # (5b) Full proximal Newton direction
        z = beta - g / H_diag
        beta_newton = soft_threshold(z, gamma / H_diag)
        direction = beta_newton - beta # here dif with proximal gradient descent since dont commit to this yet 

        # (5c) Backtracking line search (Armijo condition)
        # Descent measure: directional derivative approximation using the
        # composite gradient mapping  d = beta_newton - beta.
        # The reference decrease is  g^T d + 0.5 * d^T H d  (quadratic model decrease).
        grad_dot_d = np.dot(g, direction)
        quad_term = 0.5 * np.dot(H_diag * direction, direction)
        model_decrease = grad_dot_d - quad_term

        alpha = alpha_init
        ls_iter = 0
        max_ls = 30 # safeguard since after 30 steps shrank by almost 0 (0.5^(30))
        while ls_iter < max_ls:
            beta_trial = beta + alpha * direction
            C_trial = build_cost(beta_trial, D)
            u_t, v_t = u.copy(), v.copy()
            for _ in range(sinkhorn_inner):
                u_t, v_t = sinkhorn_one_pass(p, q, C_trial, u_t, v_t)
            pi_trial = compute_pi(u_t, v_t, C_trial)
            f_trial = sista_objective(hat_pi, pi_trial, beta_trial, gamma)

            if f_trial <= f_cur + armijo_c * alpha * model_decrease:
                break
            alpha *= armijo_rho # if fails multiply alpha by 0.5 by default and try again
            ls_iter += 1

        ls_counts.append(ls_iter)
        beta = beta_trial
        u, v = u_t, v_t

        # stopping criterion
        rel = np.linalg.norm(beta - prev_beta) / (np.linalg.norm(prev_beta) + 1e-12)
        if rel < tol:
            break
        prev_beta = beta.copy()

    return beta, u, v, history, ls_counts


# =============================================
# TEST 1
# BASIC COMPARISON: Newton vs Damped Newton
# =============================================
if __name__ == "__main__":
    import time
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    np.random.seed(42)

    N = 15
    K = 8
    gamma = 1e-3
    n_iters = 10000
    sinkhorn_inner = 10

    D = np.random.rand(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    true_beta = np.array([0.0, 1.2, 0.0, 0.0, -0.8, 0.0, 0.5, 0.0])

    C_true = build_cost(true_beta, D)
    hat_pi = np.exp(-C_true)
    hat_pi = hat_pi / hat_pi.sum()
    p = hat_pi.sum(axis=1)
    q = hat_pi.sum(axis=0)

    # --- Run Newton SISTA ---
    t0 = time.time()
    beta_nw, _, _, hist_nw = sista_newton(
        p, q, hat_pi, D, gamma=gamma,
        n_iters=n_iters, sinkhorn_inner=sinkhorn_inner, tol=1e-12
    )
    time_nw = time.time() - t0

    # --- Run Damped Newton SISTA ---
    t0 = time.time()
    beta_dnw, _, _, hist_dnw, ls_dnw = sista_damped_newton(
        p, q, hat_pi, D, gamma=gamma,
        n_iters=n_iters, sinkhorn_inner=sinkhorn_inner, tol=1e-12
    )
    time_dnw = time.time() - t0

    # =============================================
    # Print results
    # =============================================
    print("=" * 65)
    print("SISTA COMPARISON: Newton vs Damped Newton")
    print("=" * 65)
    header = f"{'':>6} {'True':>8} {'Newton':>10} {'Damped':>10}"
    print(header)
    print("-" * 65)
    for k in range(K):
        print(f"β_{k:<4} {true_beta[k]:>8.4f} {beta_nw[k]:>10.4f} {beta_dnw[k]:>10.4f}")

    print(f"\nIterations:    Newton = {len(hist_nw):<6}  Damped = {len(hist_dnw)}")
    print(f"Time:          Newton = {time_nw:.4f}s   Damped = {time_dnw:.4f}s")
    print(f"Final obj:     Newton = {hist_nw[-1]:.6f}  Damped = {hist_dnw[-1]:.6f}")

    sparsity_nw = np.sum(np.abs(beta_nw) < 1e-4)
    sparsity_dnw = np.sum(np.abs(beta_dnw) < 1e-4)
    print(f"Zeros found:   Newton = {sparsity_nw}/{K}       Damped = {sparsity_dnw}/{K}")
    print(f"||β - β_true||: Newton = {np.linalg.norm(beta_nw - true_beta):.6f}"
          f"  Damped = {np.linalg.norm(beta_dnw - true_beta):.6f}")

    avg_ls = np.mean(ls_dnw) if ls_dnw else 0
    max_ls = max(ls_dnw) if ls_dnw else 0
    full_steps = sum(1 for c in ls_dnw if c == 0)
    print(f"\nLine search stats (Damped Newton):")
    print(f"  Avg backtracks/iter: {avg_ls:.2f}")
    print(f"  Max backtracks:      {max_ls}")
    print(f"  Full Newton steps:   {full_steps}/{len(ls_dnw)} "
          f"({100*full_steps/max(len(ls_dnw),1):.0f}%)")

    # =============================================
    # Plot 1: Convergence + Plot 2: Coefficients
    # =============================================
    opt_val = min(hist_nw[-1], hist_dnw[-1])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Convergence (log scale)", "Recovered Coefficients"],
        horizontal_spacing=0.12,
    )

    for name, hist, color in [
        ("Newton",        hist_nw,  "#4ECDC4"),
        ("Damped Newton", hist_dnw, "#FF6B35"),
    ]:
        subopt = np.array(hist) - opt_val + 1e-14
        fig.add_trace(go.Scatter(
            x=list(range(len(hist))),
            y=subopt,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
        ), row=1, col=1)

    labels = [f"β_{k}" for k in range(K)]

    fig.add_trace(go.Bar(
        y=labels, x=true_beta, orientation="h",
        name="True β", marker_color="rgba(255,255,255,0.15)",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=beta_nw, orientation="h",
        name="Newton", marker_color="#4ECDC4",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=beta_dnw, orientation="h",
        name="Damped Newton", marker_color="#FF6B35",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"SISTA: Newton vs Damped Newton  (K={K}, N={N}, γ={gamma})",
            font=dict(size=18, color="#F5F5F0", family="Georgia, serif"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0B0C10",
        plot_bgcolor="#0B0C10",
        font=dict(family="Georgia, serif", color="#E8E8E0"),
        barmode="group",
        height=520,
        width=1100,
        legend=dict(orientation="h", y=-0.12, x=0.0, font=dict(size=12)),
    )

    fig.update_yaxes(row=1, col=1, title="F(β_k) − F*", type="log",
                     gridcolor="rgba(50,50,50,0.3)")
    fig.update_xaxes(row=1, col=1, title="Iteration",
                     gridcolor="rgba(50,50,50,0.3)")
    fig.update_xaxes(row=1, col=2, title="Coefficient value",
                     gridcolor="rgba(50,50,50,0.3)", zeroline=True,
                     zerolinecolor="rgba(100,100,100,0.5)")
    fig.update_yaxes(row=1, col=2, gridcolor="rgba(50,50,50,0.1)")

    fig.show()

    # =============================================
    # Helper to run one comparison
    # =============================================
    def run_comparison(label, N_t, K_t, gamma_t, sinkhorn_inner_t, n_iters_t, seed):
        np.random.seed(seed)

        D_t = np.random.randn(K_t, N_t, N_t) * 2.0
        D_t = (D_t + D_t.transpose(0, 2, 1)) / 2

        true_beta_t = np.zeros(K_t)
        true_beta_t[min(10, K_t - 1)] = 0.8
        true_beta_t[min(50, K_t - 1)] = -0.6
        true_beta_t[min(99, K_t - 1)] = 0.3
        n_true_nonzero = np.sum(np.abs(true_beta_t) > 0)

        C_t = build_cost(true_beta_t, D_t)
        hat_pi_t = np.exp(-C_t)
        hat_pi_t = hat_pi_t / hat_pi_t.sum()
        p_t = hat_pi_t.sum(axis=1)
        q_t = hat_pi_t.sum(axis=0)

        t0 = time.time()
        b_nw, _, _, h_nw = sista_newton(
            p_t, q_t, hat_pi_t, D_t, gamma=gamma_t,
            n_iters=n_iters_t, sinkhorn_inner=sinkhorn_inner_t, tol=1e-12
        )
        time_nw_t = time.time() - t0

        t0 = time.time()
        b_dnw, _, _, h_dnw, ls_dnw_t = sista_damped_newton(
            p_t, q_t, hat_pi_t, D_t, gamma=gamma_t,
            n_iters=n_iters_t, sinkhorn_inner=sinkhorn_inner_t, tol=1e-12
        )
        time_dnw_t = time.time() - t0

        print(f"\n{'=' * 65}")
        print(f"{label}")
        print(f"{'=' * 65}")
        print(f"K={K_t}, N={N_t}, γ={gamma_t}, sinkhorn_inner={sinkhorn_inner_t}")
        print(f"\n{'Metric':<25} {'Newton':>12} {'Damped':>12}")
        print("-" * 50)
        print(f"{'Iterations':<25} {len(h_nw):>12} {len(h_dnw):>12}")
        print(f"{'Wall time':<25} {time_nw_t:>11.4f}s {time_dnw_t:>11.4f}s")
        print(f"{'Final obj':<25} {h_nw[-1]:>12.6f} {h_dnw[-1]:>12.6f}")
        sp_nw = np.sum(np.abs(b_nw) < 1e-4)
        sp_dnw = np.sum(np.abs(b_dnw) < 1e-4)
        print(f"{'Zeros found':<25} {f'{sp_nw}/{K_t}':>12} {f'{sp_dnw}/{K_t}':>12}"
              f"  (true={K_t - n_true_nonzero}/{K_t})")
        print(f"{'||β - β_true||':<25} "
              f"{np.linalg.norm(b_nw - true_beta_t):>12.4f} "
              f"{np.linalg.norm(b_dnw - true_beta_t):>12.4f}")

        avg = np.mean(ls_dnw_t) if ls_dnw_t else 0
        full = sum(1 for c in ls_dnw_t if c == 0)
        print(f"{'Avg backtracks (Damped)':<25} {avg:>12.2f}")
        print(f"{'Full Newton steps':<25} {'':>12} {full}/{len(ls_dnw_t)}")

        winner = "Newton" if h_nw[-1] < h_dnw[-1] else "Damped Newton"
        print(f"\n→ Winner (lower obj): {winner}")
        return h_nw, h_dnw

    # =============================================
    # TEST 2a: Well-conditioned (many Sinkhorn iters)
    # Newton should do fine; damped should match since
    # the full step already satisfies Armijo most of the time.
    # =============================================
    h_nw_2a, h_dnw_2a = run_comparison(
        label="TEST 2a: Well-conditioned (K=8, sinkhorn_inner=10)",
        N_t=15, K_t=8, gamma_t=1e-3,
        sinkhorn_inner_t=10, n_iters_t=500, seed=42,
    )

    # =============================================
    # TEST 2b: Poorly-conditioned (few Sinkhorn iters)
    # Hessian estimates are noisy → Newton may overshoot.
    # Damped Newton should be more stable here.
    # =============================================
    h_nw_2b, h_dnw_2b = run_comparison(
        label="TEST 2b: Few Sinkhorn iters (K=8, sinkhorn_inner=1)",
        N_t=15, K_t=8, gamma_t=1e-3,
        sinkhorn_inner_t=1, n_iters_t=500, seed=42,
    )

    # =============================================
    # TEST 2c: High-dimensional (large K)
    # More coordinates → Hessian diagonal more variable.
    # =============================================
    h_nw_2c, h_dnw_2c = run_comparison(
        label="TEST 2c: High-dimensional (K=200, sinkhorn_inner=10)",
        N_t=10, K_t=200, gamma_t=1e-4,
        sinkhorn_inner_t=10, n_iters_t=200, seed=7,
    )

    # =============================================
    # TEST 2d: Stress test (large K + few Sinkhorn)
    # =============================================
    h_nw_2d, h_dnw_2d = run_comparison(
        label="TEST 2d: Stress test (K=200, sinkhorn_inner=1)",
        N_t=10, K_t=200, gamma_t=1e-4,
        sinkhorn_inner_t=1, n_iters_t=200, seed=7,
    )

    # =============================================
    # Summary plot: all 4 test convergence curves
    # =============================================
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "2a: Well-conditioned",
            "2b: Few Sinkhorn iters",
            "2c: High-dimensional",
            "2d: Stress (large K + few Sinkhorn)",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    test_data = [
        (h_nw_2a, h_dnw_2a, 1, 1),
        (h_nw_2b, h_dnw_2b, 1, 2),
        (h_nw_2c, h_dnw_2c, 2, 1),
        (h_nw_2d, h_dnw_2d, 2, 2),
    ]

    for i, (h_nw_i, h_dnw_i, row, col) in enumerate(test_data):
        opt_i = min(h_nw_i[-1], h_dnw_i[-1])
        show_legend = (i == 0)
        for name, hist, color in [
            ("Newton",        h_nw_i,  "#4ECDC4"),
            ("Damped Newton", h_dnw_i, "#FF6B35"),
        ]:
            subopt = np.array(hist) - opt_i + 1e-14
            fig2.add_trace(go.Scatter(
                x=list(range(len(hist))),
                y=subopt,
                mode="lines",
                name=name,
                showlegend=show_legend,
                line=dict(color=color, width=2),
            ), row=row, col=col)

    for r in range(1, 3):
        for c in range(1, 3):
            fig2.update_yaxes(row=r, col=c, type="log", title="F − F*",
                              gridcolor="rgba(50,50,50,0.3)")
            fig2.update_xaxes(row=r, col=c, title="Iteration",
                              gridcolor="rgba(50,50,50,0.3)")

    fig2.update_layout(
        title=dict(
            text="Newton vs Damped Newton — Convergence across test scenarios",
            font=dict(size=18, color="#F5F5F0", family="Georgia, serif"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0B0C10",
        plot_bgcolor="#0B0C10",
        font=dict(family="Georgia, serif", color="#E8E8E0"),
        height=700,
        width=1100,
        legend=dict(orientation="h", y=-0.06, x=0.0, font=dict(size=12)),
    )
    fig2.show()
