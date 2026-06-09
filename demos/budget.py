"""
Demo 4 — Budget calculator.
Given (q0, q1, p, target CI width), output minimum calibration size m
for equal split vs adaptive allocation. Show savings in human labels.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import (
    min_calibration_for_width, adaptive_allocation,
    ci_length_equal_split, bias_crossover_threshold, point_estimator,
)


def render():
    st.header("Demo 4 — Budget Calculator")
    st.markdown(
        """
        LLM judgments are cheap — you can run millions. The real cost is **human labels**
        for the calibration set. How many do you actually need to get a tight confidence
        interval? And can smart allocation save you labels?
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        q0 = st.slider("Judge specificity q0", 0.51, 1.0, 0.70, 0.01, key="bud_q0")
        q1 = st.slider("Judge sensitivity q1", 0.51, 1.0, 0.90, 0.01, key="bud_q1")
        p  = st.slider("Observed naive score p̂", 0.05, 0.95, 0.50, 0.01, key="bud_p")
    with col2:
        target_width = st.slider(
            "Target CI full-width (e.g. 0.10 = ±0.05)",
            0.04, 0.40, 0.10, 0.02, key="bud_width",
        )
        alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05, key="bud_alpha")

    if q0 + q1 <= 1.0:
        st.error("q0 + q1 must be > 1.")
        return

    theta_hat = point_estimator(p, q0, q1)

    m_equal    = min_calibration_for_width(p, q0, q1, target_width, alpha, adaptive=False)
    m_adaptive = min_calibration_for_width(p, q0, q1, target_width, alpha, adaptive=True)
    m0_opt, m1_opt, _ = adaptive_allocation(p, q0, q1, m_adaptive, alpha)

    savings    = max(0, m_equal - m_adaptive)
    pct_saving = savings / max(m_equal, 1) * 100

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Corrected estimate θ̂", f"{theta_hat:.3f}")
    c2.metric("m needed (equal split)", f"{m_equal}")
    c3.metric("m needed (adaptive)", f"{m_adaptive}",
              delta=f"−{savings} labels saved ({pct_saving:.0f}%)",
              delta_color="inverse" if savings == 0 else "normal")

    if savings > 0:
        st.success(
            f"Adaptive allocation saves **{savings} human labels** ({pct_saving:.0f}%) "
            f"by using m0 ≈ {int(m0_opt)} and m1 ≈ {int(m1_opt)} instead of an equal split."
        )
    else:
        st.info("Equal and adaptive splits require similar budgets for these parameters.")

    # CI length vs m curve
    st.subheader("CI width vs calibration budget")
    m_range = np.arange(10, min(m_equal * 3, 2000) + 1, 10)

    lengths_equal    = [ci_length_equal_split(p, q0, q1, m, alpha) for m in m_range]
    lengths_adaptive = []
    for m in m_range:
        _, _, l = adaptive_allocation(p, q0, q1, m, alpha)
        lengths_adaptive.append(l)

    fig = go.Figure()
    fig.add_hline(y=target_width, line_dash="dash", line_color="gray",
                  annotation_text=f"Target width = {target_width}", annotation_position="right")
    fig.add_trace(go.Scatter(
        x=m_range, y=lengths_equal,
        mode="lines", name="Equal split (m0 = m1)",
        line=dict(color="#636EFA", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=m_range, y=lengths_adaptive,
        mode="lines", name="Adaptive allocation",
        line=dict(color="#00CC96", width=2),
    ))
    fig.add_vline(x=m_equal, line_dash="dot", line_color="#636EFA",
                  annotation_text=f"m={m_equal}", annotation_position="top right")
    fig.add_vline(x=m_adaptive, line_dash="dot", line_color="#00CC96",
                  annotation_text=f"m={m_adaptive}", annotation_position="top left")
    fig.update_layout(
        xaxis_title="Total calibration size m = m0 + m1",
        yaxis_title="95% CI full-width",
        yaxis=dict(range=[0, max(lengths_equal[0], target_width * 2)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How does adaptive allocation work?"):
        theta_star = bias_crossover_threshold(q0, q1)
        kappa = (1 - q0) / max(1 - q1, 1e-9)
        ratio = ((1 / max(p, 1e-9)) - 1) * (kappa ** 0.5)
        st.markdown(f"""
The two calibration label types (truly-wrong examples = m0, truly-correct = m1)
contribute **asymmetrically** to the total variance. The optimal ratio is:

$$\\frac{{m_0}}{{m_1}} \\approx \\left(\\frac{{1}}{{\\hat{{p}}}} - 1\\right) \\sqrt{{\\kappa}}, \\quad \\kappa = \\frac{{1-q_0}}{{1-q_1}}$$

With your settings: κ = {kappa:.3f}, optimal m0/m1 ≈ {ratio:.2f}
→ m0 ≈ **{int(m0_opt)}**, m1 ≈ **{int(m1_opt)}** out of total m = {m_adaptive}

The intuition: if the judge makes more false-positive errors (low q0), you need
more m0 examples to pin down q0 precisely. If you have a high naive score p̂,
most of the test-set responses are "correct", so m1 errors hurt more.
        """)
