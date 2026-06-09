"""
Demo 5 — LLM-vs-human map.
For a symmetric judge (q0 = q1 = q), show the region in (q, theta) space
where calibrated LLM evaluation beats human-only evaluation at equal budget.
The boundary appears only above q ≈ 0.854.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import llm_beats_human_region
from math import sqrt


_THRESHOLD = 0.5 + 1 / (2 * sqrt(2))  # ≈ 0.854


def render():
    st.header("Demo 5 — When Does an LLM Judge Actually Help?")
    st.markdown(
        f"""
        Surprising result from the paper: calibrated LLM evaluation **beats** pure human
        evaluation only when judge quality `q > {_THRESHOLD:.3f}` — and even then, only
        for a specific band of true accuracy values.

        Below that threshold, you'd be better off spending your entire budget on human labels.
        """
    )

    q_vals     = np.linspace(0.5, 1.0, 300)
    theta_lo   = []
    theta_hi   = []
    q_region   = []

    for q in q_vals:
        result = llm_beats_human_region(q)
        if result is not None:
            lo, hi = result
            q_region.append(q)
            theta_lo.append(lo)
            theta_hi.append(hi)

    q_region = np.array(q_region)
    theta_lo = np.array(theta_lo)
    theta_hi = np.array(theta_hi)

    fig = go.Figure()

    # Shaded "LLM wins" region
    if len(q_region) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([q_region, q_region[::-1]]),
            y=np.concatenate([theta_hi, theta_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="LLM beats human (calibrated)",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=q_region, y=theta_hi,
            mode="lines", name="Upper bound",
            line=dict(color="#00CC96", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=q_region, y=theta_lo,
            mode="lines", name="Lower bound",
            line=dict(color="#00CC96", width=2),
        ))

    # Threshold line
    fig.add_vline(
        x=_THRESHOLD,
        line_dash="dash",
        line_color="#EF553B",
        annotation_text=f"q* ≈ {_THRESHOLD:.3f}",
        annotation_position="top right",
    )

    # y = 0.5 reference
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)

    # Interactive point
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        q_user = st.slider("Your judge quality q", 0.50, 1.0, 0.88, 0.01, key="lvh_q")
    with col2:
        theta_user = st.slider("True accuracy θ", 0.0, 1.0, 0.60, 0.01, key="lvh_theta")

    region = llm_beats_human_region(q_user)
    if region is not None:
        lo_u, hi_u = region
        in_region = lo_u <= theta_user <= hi_u
        verdict = (
            f"✅ LLM evaluation **helps** here (θ in [{lo_u:.2f}, {hi_u:.2f}])"
            if in_region
            else f"❌ LLM evaluation **does not help** here — θ is outside [{lo_u:.2f}, {hi_u:.2f}]"
        )
    else:
        verdict = f"❌ Judge quality q = {q_user:.2f} is below threshold q* ≈ {_THRESHOLD:.3f} — pure human labels win."

    # Add user point to chart
    fig.add_trace(go.Scatter(
        x=[q_user], y=[theta_user],
        mode="markers",
        name="Your setting",
        marker=dict(symbol="circle", size=14, color="#B6005E",
                    line=dict(color="black", width=1.5)),
    ))

    fig.update_layout(
        xaxis_title="Judge quality q (symmetric: q0 = q1 = q)",
        yaxis_title="True accuracy θ",
        xaxis=dict(range=[0.5, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(verdict)

    with st.expander("The math behind the boundary"):
        st.markdown(f"""
For a **symmetric** judge (q0 = q1 = q) and a fixed total budget split between
human labels (calibration) and LLM calls (test set), the calibrated LLM estimator
has lower variance than pure human evaluation only when:

$$q > \\frac{{1}}{{2}} + \\frac{{1}}{{2\\sqrt{{2}}}} \\approx {_THRESHOLD:.4f}$$

When this holds, the advantage exists for θ in:

$$\\left[\\frac{{1}}{{2}} - \\sqrt{{\\frac{{1}}{{2}} - \\frac{{1}}{{4(2q-1)^2}}}},\\;
         \\frac{{1}}{{2}} + \\sqrt{{\\frac{{1}}{{2}} - \\frac{{1}}{{4(2q-1)^2}}}}\\right]$$

Outside this band (very high or very low θ) the judge's errors dominate and
human-only evaluation wins regardless of q.
        """)
