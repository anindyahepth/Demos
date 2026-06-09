"""
Demo 2 — Interactive bias curve.
Sliders for q0, q1; live plot of E[p_hat] vs theta with y=x diagonal
and the theta* crossover marker.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import expected_naive_score, bias_crossover_threshold


def render():
    st.header("Demo 2 — The Bias Curve")
    st.markdown(
        """
        The LLM judge's expected score **E[p̂]** is a linear function of true accuracy **θ**.
        It crosses the diagonal (unbiased line) at exactly one point: the **bias crossover
        threshold θ\\***. Below θ\\* the judge *over*estimates; above it the judge *under*estimates.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        q0 = st.slider("Specificity q0  (P(judge=wrong | truth=wrong))", 0.51, 1.0, 0.70, 0.01, key="bc_q0")
    with col2:
        q1 = st.slider("Sensitivity q1  (P(judge=right | truth=right))", 0.51, 1.0, 0.90, 0.01, key="bc_q1")

    if q0 + q1 <= 1.0:
        st.error("q0 + q1 must be > 1. Increase either slider.")
        return

    theta_vals = np.linspace(0, 1, 300)
    e_p = expected_naive_score(theta_vals, q0, q1)
    theta_star = bias_crossover_threshold(q0, q1)
    e_p_star = expected_naive_score(theta_star, q0, q1)

    overestimate_mask  = theta_vals <= theta_star
    underestimate_mask = theta_vals >= theta_star

    fig = go.Figure()

    # y = x diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Unbiased (y = θ)",
        line=dict(color="black", dash="dash", width=1.5),
    ))

    # Overestimation region (below theta*)
    fig.add_trace(go.Scatter(
        x=theta_vals[overestimate_mask],
        y=e_p[overestimate_mask],
        mode="lines",
        name="E[p̂] — overestimates θ",
        line=dict(color="#EF553B", width=3),
    ))

    # Underestimation region (above theta*)
    fig.add_trace(go.Scatter(
        x=theta_vals[underestimate_mask],
        y=e_p[underestimate_mask],
        mode="lines",
        name="E[p̂] — underestimates θ",
        line=dict(color="#636EFA", width=3),
    ))

    # theta* crossover marker
    fig.add_trace(go.Scatter(
        x=[theta_star],
        y=[e_p_star],
        mode="markers+text",
        name=f"θ* = {theta_star:.3f}",
        marker=dict(symbol="star", size=16, color="gold", line=dict(color="black", width=1)),
        text=[f"θ* = {theta_star:.3f}"],
        textposition="top right",
    ))

    # Vertical reference line at theta*
    fig.add_vline(
        x=theta_star,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"θ* = {theta_star:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="True accuracy θ",
        yaxis_title="Expected naive judge score E[p̂]",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Numeric summary
    bias_at_low  = expected_naive_score(0.3, q0, q1) - 0.3
    bias_at_high = expected_naive_score(0.8, q0, q1) - 0.8

    c1, c2, c3 = st.columns(3)
    c1.metric("Bias crossover θ*", f"{theta_star:.3f}")
    c2.metric("Bias at θ = 0.3", f"{bias_at_low:+.3f}", delta_color="inverse")
    c3.metric("Bias at θ = 0.8", f"{bias_at_high:+.3f}", delta_color="inverse")

    with st.expander("The math"):
        st.markdown(f"""
The expected naive score follows directly from the confusion matrix:

$$E[\\hat{{p}}] = (q_0 + q_1 - 1)\\,\\theta + (1 - q_0)$$

Setting $E[\\hat{{p}}] = \\theta$ and solving gives the crossover:

$$\\theta^* = \\frac{{1 - q_0}}{{2 - q_0 - q_1}} = {theta_star:.4f}$$

With your current settings (q0 = {q0}, q1 = {q1}):
- The slope of E[p̂] vs θ is **{q0 + q1 - 1:.3f}** (< 1, so the curve is flatter than the diagonal)
- The judge has false-positive rate **{1-q0:.2f}** (marks wrong answers as correct)
- The judge has false-negative rate **{1-q1:.2f}** (marks correct answers as wrong)
        """)
