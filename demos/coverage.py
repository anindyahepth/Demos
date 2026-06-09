"""
Demo 3 — Coverage gut-punch.
Monte Carlo: sweep true theta, compare naive CI coverage vs corrected CI coverage.
Naive CI targets p_hat (not theta) — its coverage of theta collapses to ~0 except at theta*.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import (
    confidence_interval, naive_ci, bias_crossover_threshold,
    expected_naive_score,
)


def _simulate_coverage(theta_vals, q0, q1, n, m0, m1, n_trials, alpha, rng):
    naive_cover  = np.zeros(len(theta_vals))
    corr_cover   = np.zeros(len(theta_vals))

    for i, theta in enumerate(theta_vals):
        naive_hits = 0
        corr_hits  = 0
        for _ in range(n_trials):
            # Simulate test set
            z  = rng.binomial(1, theta, n)
            z_hat = np.where(z == 1,
                             rng.binomial(1, q1, n),
                             1 - rng.binomial(1, q0, n))
            p = z_hat.mean()

            # Simulate calibration set
            z_calib1 = np.ones(m1, dtype=int)
            z_calib0 = np.zeros(m0, dtype=int)
            zh_calib1 = rng.binomial(1, q1, m1)
            zh_calib0 = 1 - rng.binomial(1, q0, m0)
            q1_hat = zh_calib1.mean() if m1 > 0 else q1
            q0_hat = (1 - zh_calib0).mean() if m0 > 0 else q0  # specificity
            # fix: specificity = fraction of z=0 correctly labeled 0
            q0_hat = zh_calib0.mean() if m0 > 0 else q0

            # Naive CI (for p, not theta)
            lo_n, hi_n = naive_ci(p, n, alpha)
            naive_hits += (lo_n <= theta <= hi_n)

            # Corrected CI
            if q0_hat + q1_hat > 1.0:
                lo_c, hi_c = confidence_interval(p, q0_hat, q1_hat, n, m0, m1, alpha)
                corr_hits += (lo_c <= theta <= hi_c)
            # if judge looks worse than random on this trial, skip (conservative)

        naive_cover[i] = naive_hits / n_trials
        corr_cover[i]  = corr_hits  / n_trials

    return naive_cover, corr_cover


def render():
    st.header("Demo 3 — The Coverage Gut-Punch")
    st.markdown(
        """
        A 95% confidence interval should contain the true value **95% of the time**.
        The naive CI (`p̂ ± 1.96·SE`) targets `p̂`, not `θ` — so its actual coverage of
        the true accuracy θ is near **zero** almost everywhere.
        Only right at the bias crossover θ\\* does it accidentally hit 95%.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        q0 = st.slider("Specificity q0", 0.51, 1.0, 0.70, 0.01, key="cov_q0")
        q1 = st.slider("Sensitivity q1", 0.51, 1.0, 0.90, 0.01, key="cov_q1")
    with col2:
        n    = st.select_slider("Test-set size n", [100, 500, 1000, 5000], value=500, key="cov_n")
        m_eq = st.select_slider("Calibration size m (equal split)", [20, 50, 100, 200], value=50, key="cov_m")
    with col3:
        n_trials = st.select_slider("Monte Carlo trials", [200, 500, 1000], value=500, key="cov_trials")
        alpha    = st.select_slider("α (CI level = 1−α)", [0.01, 0.05, 0.10], value=0.05, key="cov_alpha")

    if q0 + q1 <= 1.0:
        st.error("q0 + q1 must be > 1.")
        return

    m0 = m1 = m_eq // 2

    if st.button("Run simulation", key="cov_run"):
        with st.spinner("Simulating…"):
            rng = np.random.default_rng(42)
            theta_vals = np.linspace(0.05, 0.95, 40)
            naive_cov, corr_cov = _simulate_coverage(
                theta_vals, q0, q1, n, m0, m1, n_trials, alpha, rng
            )

        theta_star = bias_crossover_threshold(q0, q1)
        target = 1 - alpha

        fig = go.Figure()
        fig.add_hline(y=target, line_dash="dash", line_color="gray",
                      annotation_text=f"Target {target:.0%}", annotation_position="right")
        fig.add_trace(go.Scatter(
            x=theta_vals, y=naive_cov,
            mode="lines+markers", name="Naive CI coverage",
            line=dict(color="#EF553B", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=theta_vals, y=corr_cov,
            mode="lines+markers", name="Corrected CI coverage",
            line=dict(color="#00CC96", width=2),
        ))
        fig.add_vline(x=theta_star, line_dash="dot", line_color="gray",
                      annotation_text=f"θ* = {theta_star:.2f}", annotation_position="top right")
        fig.update_layout(
            xaxis_title="True accuracy θ",
            yaxis_title="Empirical coverage",
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        mean_naive = naive_cov.mean()
        mean_corr  = corr_cov.mean()
        c1, c2 = st.columns(2)
        c1.metric("Mean naive coverage", f"{mean_naive:.1%}", delta=f"{mean_naive - target:+.1%}", delta_color="inverse")
        c2.metric("Mean corrected coverage", f"{mean_corr:.1%}", delta=f"{mean_corr - target:+.1%}")

    else:
        st.info("Set parameters above, then click **Run simulation**.")

    with st.expander("Why is the naive CI so wrong?"):
        st.markdown("""
The naive CI is `p̂ ± z · SE(p̂)`. This interval is perfectly valid — for **p̂**.
But p̂ is a biased estimate of θ, so the interval is centered in the wrong place.
It covers the *judge's expected score*, not the *true accuracy*.

The corrected CI accounts for:
1. **Sampling variance** of p̂ on the test set (n)
2. **Estimation uncertainty** of q̂0 on the calibration set (m0)
3. **Estimation uncertainty** of q̂1 on the calibration set (m1)

All three sources propagate through the bias-correction formula.
        """)
