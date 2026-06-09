"""
Demo 1 — Leaderboard that lies.
Two models with different judge profiles; naive score can reverse the ranking.
Per-model calibration restores the true ordering.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import point_estimator, expected_naive_score, bias_crossover_threshold, _check_judge


def render():
    st.header("Demo 1 — The Leaderboard That Lies")
    st.markdown(
        """
        Two models are evaluated with the **same LLM judge** — but that judge may behave
        differently on each model's outputs (different topics, styles, or difficulty levels).
        Adjust the sliders below and watch the leaderboard ranking flip.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model A")
        theta_a = st.slider("True accuracy θ_A", 0.0, 1.0, 0.72, 0.01, key="ta")
        q0_a = st.slider("Judge specificity q0_A", 0.5, 1.0, 0.65, 0.01, key="q0a")
        q1_a = st.slider("Judge sensitivity q1_A", 0.5, 1.0, 0.90, 0.01, key="q1a")

    with col2:
        st.subheader("Model B")
        theta_b = st.slider("True accuracy θ_B", 0.0, 1.0, 0.68, 0.01, key="tb")
        q0_b = st.slider("Judge specificity q0_B", 0.5, 1.0, 0.85, 0.01, key="q0b")
        q1_b = st.slider("Judge sensitivity q1_B", 0.5, 1.0, 0.80, 0.01, key="q1b")

    # Validate
    valid_a = q0_a + q1_a > 1
    valid_b = q0_b + q1_b > 1
    if not valid_a:
        st.error("Model A: q0 + q1 must be > 1 for a useful judge.")
        return
    if not valid_b:
        st.error("Model B: q0 + q1 must be > 1 for a useful judge.")
        return

    # Compute scores
    p_a = expected_naive_score(theta_a, q0_a, q1_a)
    p_b = expected_naive_score(theta_b, q0_b, q1_b)
    corr_a = point_estimator(p_a, q0_a, q1_a)
    corr_b = point_estimator(p_b, q0_b, q1_b)

    naive_winner  = "A" if p_a > p_b else ("B" if p_b > p_a else "tie")
    true_winner   = "A" if theta_a > theta_b else ("B" if theta_b > theta_a else "tie")
    corr_winner   = "A" if corr_a > corr_b else ("B" if corr_b > corr_a else "tie")
    ranking_flips = naive_winner != true_winner

    # Chart
    categories = ["Model A", "Model B"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Naive judge score p̂",
        x=categories,
        y=[p_a, p_b],
        marker_color=["#EF553B", "#636EFA"],
        opacity=0.6,
    ))
    fig.add_trace(go.Bar(
        name="Corrected estimate θ̂",
        x=categories,
        y=[corr_a, corr_b],
        marker_color=["#EF553B", "#636EFA"],
    ))
    fig.add_trace(go.Scatter(
        name="True accuracy θ",
        x=categories,
        y=[theta_a, theta_b],
        mode="markers",
        marker=dict(symbol="diamond", size=14, color="black"),
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1]),
        xaxis_title="Model",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Verdict
    if ranking_flips:
        st.error(
            f"🚨 **Ranking flip!** Naive judge says **Model {naive_winner} wins**, "
            f"but the true winner is **Model {true_winner}**. "
            f"Corrected estimate agrees with truth: **Model {corr_winner}**."
        )
    else:
        st.success(
            f"✅ Rankings agree in this configuration — naive judge says **{naive_winner}**, "
            f"truth is **{true_winner}**. Try shifting the sliders to trigger a flip."
        )

    # Explain the mechanism
    with st.expander("Why does this happen?"):
        star_a = bias_crossover_threshold(q0_a, q1_a)
        star_b = bias_crossover_threshold(q0_b, q1_b)
        bias_a = p_a - theta_a
        bias_b = p_b - theta_b
        st.markdown(f"""
The naive score is **not** an unbiased estimator of true accuracy. Its expected value is:

$$E[\\hat{{p}}] = (q_0 + q_1 - 1)\\,\\theta + (1 - q_0)$$

This means:
- **Model A**: bias crossover θ\\* = **{star_a:.2f}** → bias = **{bias_a:+.3f}**
- **Model B**: bias crossover θ\\* = **{star_b:.2f}** → bias = **{bias_b:+.3f}**

Each model's judge inflates or deflates its score by a *different* amount, making
naive comparisons unreliable even when both judges are "decent".
        """)
