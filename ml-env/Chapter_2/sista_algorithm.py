"""
Created 21/02/2026
This file implements the SISTA algorithm with proximal gradient descent as well as with the Newton Method instead of proximal gradient descent
which allows us to compare both methods. 

Here we can't use gradient descent because the L1 norm is not differentiable at 0 so we use ISTA (proximal gradient descent). 
SISTA alternates between: 
1. Sinkhorn step --> updating u and v 
2. ISTA step --> updating beta to move β in the direction that makes π closer to ^π, while applying an L1 sparsity penalty (soft-thresholding)
Code thus has two core parts: 
sinkhorn_one_pass(...) and beta update via gradient + soft_threshold

24/02/2026 
Add sista with newton method instead of proximal gradient descent. Components are mostly the same, step 5 is the only thing that changes. 
"""

import numpy as np

"""
Sinkhorn uses a lot of log sum of exp formulas so we first implement a utility function
Issue is that exp can overflow when inputs are too large so trick is that we substract the max of a from each explonential in the sum
"""
def logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out 

"""
Soft thresholding implementation --> its the proximal operator for lambda l1 norm, its what creates sparsity
"""
def soft_threshold(z, lam):
    return np.sign(z)*np.maximum(np.abs(z)-lam, 0.0)

"""
Cost, output is a NxN matrix 
"""
def build_cost(beta, D):
    return np.tensordot(beta,D, axes=(0,0))

"""
Sinkhorn to update u and v
"""
def sinkhorn_one_pass(p,q,C,u,v):
    u = np.log(p) - logsumexp(v[None,:] - C, axis=1)
    v = np.log(q) - logsumexp(u[:, None] - C, axis=0)
    return u, v

"""
compute pi to reconstruct the current transport plan 
"""
def compute_pi(u,v,C):
    return np.exp(u[:, None]+v[None,:]-C)


"""
Objective: KL(hat_pi || pi) + gamma * ||beta||_1
where pi is the current Sinkhorn plan for cost C_beta
"""
def sista_objective(hat_pi, pi, beta, gamma):
    log_ratio = np.log(hat_pi / (pi + 1e-30) + 1e-30)
    kl = np.sum(hat_pi * log_ratio)
    return kl + gamma * np.sum(np.abs(beta))


"""
MAIN SISTA LOOP - with proximal gradient descent 
"""
def sista(p, q, hat_pi, D, beta0=None, u0=None, v0=None,
          rho=1e-1, gamma=1e-2, n_iters=500,
          sinkhorn_inner=1, tol=1e-7, verbose=False):

    N = p.shape[0]
    K = D.shape[0]

    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    hat_pi = hat_pi / hat_pi.sum()

    beta = np.zeros(K) if beta0 is None else beta0.astype(float).copy()
    u = np.zeros(N) if u0 is None else u0.astype(float).copy()
    v = np.zeros(N) if v0 is None else v0.astype(float).copy()

    prev_beta = beta.copy()
    history = []

    for t in range(n_iters):
        # (1) cost from beta
        C = build_cost(beta, D)

        # (2) Sinkhorn updates: enforce marginals for current cost
        for _ in range(sinkhorn_inner):
            u, v = sinkhorn_one_pass(p, q, C, u, v)

        # (3) current plan from u,v,beta
        pi = compute_pi(u, v, C)
        history.append(sista_objective(hat_pi, pi, beta, gamma))

        # (4) gradient wrt beta_k: g_k = <hat_pi - pi, D[k]>
        diff = (hat_pi - pi)
        g = np.tensordot(D, diff, axes=([1, 2], [0, 1]))  # (K,)

        # (5) ISTA prox update
        z = beta - rho * g
        beta = soft_threshold(z, rho * gamma)

        # stopping criterion
        rel = np.linalg.norm(beta - prev_beta) / (np.linalg.norm(prev_beta) + 1e-12)
        if rel < tol:
            break
        prev_beta = beta

    return beta, u, v, history


"""
Sista with newton method 
Only step (5) changes: instead of a fixed step size rho for all coordinates,
each coordinate gets its own step size 1/H_kk from the Hessian diagonal.
"""
def sista_newton(p, q, hat_pi, D, beta0=None, u0=None, v0=None,
          gamma=1e-2, n_iters=500,
          sinkhorn_inner=1, tol=1e-7, mu=1e-6, verbose=False):

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

    for t in range(n_iters):
        # (1) cost from beta
        C = build_cost(beta, D)

        # (2) Sinkhorn updates
        for _ in range(sinkhorn_inner):
            u, v = sinkhorn_one_pass(p, q, C, u, v)

        # (3) current plan
        pi = compute_pi(u, v, C)
        history.append(sista_objective(hat_pi, pi, beta, gamma))

        # (4) gradient: g_k = <hat_pi - pi, D[k]>
        diff = (hat_pi - pi)
        g = np.tensordot(D, diff, axes=([1, 2], [0, 1]))  # (K,)

        # (5a) Hessian diagonal: H_kk = Var_pi(D[k]) = <pi, D[k]^2> - <pi, D[k]>^2
        # flattening to allow to compute expectation as a dot product instead of double sum
        D_flat = D.reshape(K, -1)         # (K, N*N)
        pi_flat = pi.ravel()              # (N*N,)
        mean_k = D_flat @ pi_flat         # (K,)  E_pi[D_k]
        mean_sq_k = (D_flat**2) @ pi_flat # (K,)  E_pi[D_k^2]
        H_diag = mean_sq_k - mean_k**2 + mu  # (K,) add mu for numerical stability

        # (5b) Proximal Newton update (coordinate-wise)
        """
        If H_kk is large (high curvature) then small step, small threshold since need to be careful 
        If H_kk small (flat curvature) then big step and bog threshold since safe to move aggressively 
        """
        z = beta - g / H_diag
        beta = soft_threshold(z, gamma / H_diag)

        # stopping criterion
        rel = np.linalg.norm(beta - prev_beta) / (np.linalg.norm(prev_beta) + 1e-12)
        if rel < tol:
            break
        prev_beta = beta

    return beta, u, v, history


# =============================================
# TEST 1
# BASIC COMPARISON: Proximal Gradient vs Newton 
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

    # Random dictionary of N x N cost matrices
    D = np.random.rand(K, N, N) * 0.5
    D = (D + D.transpose(0, 2, 1)) / 2

    # Ground truth: sparse beta (only 3 out of 8 are nonzero)
    true_beta = np.array([0.0, 1.2, 0.0, 0.0, -0.8, 0.0, 0.5, 0.0])

    # Build ground truth cost and plan
    C_true = build_cost(true_beta, D)
    hat_pi = np.exp(-C_true)
    hat_pi = hat_pi / hat_pi.sum()
    p = hat_pi.sum(axis=1)
    q = hat_pi.sum(axis=0)

    # --- Run proximal gradient SISTA ---
    t0 = time.time()
    beta_pg, _, _, hist_pg = sista(
        p, q, hat_pi, D, gamma=gamma,
        rho=0.5, n_iters=n_iters, sinkhorn_inner=sinkhorn_inner, tol=1e-12
    )
    time_pg = time.time() - t0

    # --- Run Newton SISTA ---
    t0 = time.time()
    beta_nw, _, _, hist_nw = sista_newton(
        p, q, hat_pi, D, gamma=gamma,
        n_iters=n_iters, sinkhorn_inner=sinkhorn_inner, tol=1e-12
    )
    time_nw = time.time() - t0

    # =============================================
    # Print results
    # =============================================
    print("=" * 65)
    print("SISTA COMPARISON: Proximal Gradient vs Proximal Newton")
    print("=" * 65)
    header = f"{'':>6} {'True':>8} {'ProxGrad':>10} {'Newton':>10}"
    print(header)
    print("-" * 65)
    for k in range(K):
        print(f"β_{k:<4} {true_beta[k]:>8.4f} {beta_pg[k]:>10.4f} {beta_nw[k]:>10.4f}")

    print(f"\nIterations:    ProxGrad = {len(hist_pg):<6}  Newton = {len(hist_nw)}")
    print(f"Time:     ProxGrad = {time_pg:.4f}s   Newton = {time_nw:.4f}s")
    print(f"Final obj:     ProxGrad = {hist_pg[-1]:.6f}  Newton = {hist_nw[-1]:.6f}")

    sparsity_pg = np.sum(np.abs(beta_pg) < 1e-4)
    sparsity_nw = np.sum(np.abs(beta_nw) < 1e-4)
    print(f"Zeros found:   ProxGrad = {sparsity_pg}/{K}       Newton = {sparsity_nw}/{K}")
    print(f"||β - β_true||: ProxGrad = {np.linalg.norm(beta_pg - true_beta):.6f}"
          f"  Newton = {np.linalg.norm(beta_nw - true_beta):.6f}")

    # =============================================
    # Plot 1: Convergence + Plot 2: Coefficients
    # =============================================
    opt_val = min(hist_pg[-1], hist_nw[-1])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Convergence (log scale)", "Recovered Coefficients"],
        horizontal_spacing=0.12,
    )

    for name, hist, color, dash in [
        ("Proximal Gradient", hist_pg, "#FF6B35", "solid"),
        ("Proximal Newton",   hist_nw, "#4ECDC4", "solid"),
    ]:
        subopt = np.array(hist) - opt_val + 1e-14
        fig.add_trace(go.Scatter(
            x=list(range(len(hist))),
            y=subopt,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5, dash=dash),
        ), row=1, col=1)

    labels = [f"β_{k}" for k in range(K)]

    fig.add_trace(go.Bar(
        y=labels, x=true_beta, orientation="h",
        name="True β", marker_color="rgba(255,255,255,0.15)",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=beta_pg, orientation="h",
        name="ProxGrad", marker_color="#FF6B35",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=beta_nw, orientation="h",
        name="Newton", marker_color="#4ECDC4",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"SISTA: Proximal Gradient vs Newton  (K={K}, N={N}, γ={gamma})",
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
    # Helper to run one comparison and print results
    # =============================================
    def run_comparison(label, N_t, K_t, gamma_t, rho_t, sinkhorn_inner_t, n_iters_t, seed):
        np.random.seed(seed)

        D_t = np.random.randn(K_t, N_t, N_t) * 2.0
        D_t = (D_t + D_t.transpose(0, 2, 1)) / 2

        true_beta_t = np.zeros(K_t)
        true_beta_t[min(10, K_t-1)] = 0.8
        true_beta_t[min(50, K_t-1)] = -0.6
        true_beta_t[min(99, K_t-1)] = 0.3
        n_true_nonzero = np.sum(np.abs(true_beta_t) > 0)

        C_t = build_cost(true_beta_t, D_t)
        hat_pi_t = np.exp(-C_t)
        hat_pi_t = hat_pi_t / hat_pi_t.sum()
        p_t = hat_pi_t.sum(axis=1)
        q_t = hat_pi_t.sum(axis=0)

        t0 = time.time()
        b_pg, _, _, h_pg = sista(
            p_t, q_t, hat_pi_t, D_t, gamma=gamma_t,
            rho=rho_t, n_iters=n_iters_t, sinkhorn_inner=sinkhorn_inner_t, tol=1e-12
        )
        time_pg_t = time.time() - t0

        t0 = time.time()
        b_nw, _, _, h_nw = sista_newton(
            p_t, q_t, hat_pi_t, D_t, gamma=gamma_t,
            n_iters=n_iters_t, sinkhorn_inner=sinkhorn_inner_t, tol=1e-12
        )
        time_nw_t = time.time() - t0

        print(f"\n{'=' * 65}")
        print(f"{label}")
        print(f"{'=' * 65}")
        print(f"K={K_t}, N={N_t}, γ={gamma_t}, sinkhorn_inner={sinkhorn_inner_t}, rho={rho_t}")
        print(f"\n{'Metric':<20} {'ProxGrad':>12} {'Newton':>12}")
        print("-" * 45)
        print(f"{'Iterations':<20} {len(h_pg):>12} {len(h_nw):>12}")
        print(f"{'Wall time':<20} {time_pg_t:>11.4f}s {time_nw_t:>11.4f}s")
        print(f"{'Final obj':<20} {h_pg[-1]:>12.6f} {h_nw[-1]:>12.6f}")
        sp_pg = np.sum(np.abs(b_pg) < 1e-4)
        sp_nw = np.sum(np.abs(b_nw) < 1e-4)
        print(f"{'Zeros found':<20} {f'{sp_pg}/{K_t}':>12} {f'{sp_nw}/{K_t}':>12}  (true={K_t - n_true_nonzero}/{K_t})")
        print(f"{'||β - β_true||':<20} {np.linalg.norm(b_pg - true_beta_t):>12.4f} {np.linalg.norm(b_nw - true_beta_t):>12.4f}")

        winner = "ProxGrad" if h_pg[-1] < h_nw[-1] else "Newton"
        print(f"\n→ Winner: {winner}")

    # =============================================
    # TEST 2a: Only change = few Sinkhorn iterations (K stays small)
    # Isolates the effect of noisy π / Hessian
    # =============================================
    run_comparison(
        label="TEST 2a: Few Sinkhorn iterations (K=8, sinkhorn_inner=1)",
        N_t=15, K_t=8, gamma_t=1e-3, rho_t=0.5,
        sinkhorn_inner_t=1, n_iters_t=500, seed=42,
    )

    # =============================================
    # TEST 2b: Only change = large K (Sinkhorn stays well-converged)
    # Isolates the effect of high dimensionality
    # =============================================
    run_comparison(
        label="TEST 2b: Large K (K=200, sinkhorn_inner=10)",
        N_t=10, K_t=200, gamma_t=1e-4, rho_t=0.01,
        sinkhorn_inner_t=10, n_iters_t=200, seed=7,
    )

    # =============================================
    # TEST 2c: Both together (large K + few Sinkhorn)
    # =============================================
    run_comparison(
        label="TEST 2c: Both (K=200, sinkhorn_inner=1)",
        N_t=10, K_t=200, gamma_t=1e-4, rho_t=0.01,
        sinkhorn_inner_t=1, n_iters_t=200, seed=7,
    )
