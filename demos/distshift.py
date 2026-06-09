"""
Demo 6 — Distribution-shift breaker.
Fix test prevalence; sweep calibration prevalence.
Plot bias of four estimators — only ours stays flat at zero.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from judge_stats import (
    point_estimator, expected_naive_score, bias_crossover_threshold,
)


def _compute_biases(theta_test, q0, q1, calib_prevs):
    """
    For each calibration prevalence, compute the bias of four estimators.
    We assume n -> inf so sample variance is zero; focus on systematic bias.

    theta_calib = calib_prev  (true accuracy in calibration set)
    p_test      = E[p_hat | theta_test]   (expected naive score on test)
    p_calib     = E[p_hat | theta_calib]  (expected naive score on calib)
    """
    p_test = expected_naive_score(theta_test, q0, q1)

    naive_bias   = []
    ppi_bias     = []
    calib_bias   = []
    ours_bias    = []

    for theta_calib in calib_prevs:
        p_calib = expected_naive_score(theta_calib, q0, q1)

        # Naive: just report p_test
        naive_bias.append(p_test - theta_test)

        # PPI / difference: p_test - p_calib + theta_calib
        ppi_est = p_test - p_calib + theta_calib
        ppi_bias.append(ppi_est - theta_test)

        # Calibration-only: use theta_calib as the estimate of theta_test
        calib_bias.append(theta_calib - theta_test)

        # Ours: bias-corrected — unbiased regardless of calibration prevalence
        # because it only depends on P(Z_hat | Z), not P(Z)
        our_est = point_estimator(p_test, q0, q1)
        ours_bias.append(our_est - theta_test)

    return (
        np.array(naive_bias),
        np.array(ppi_bias),
        np.array(calib_bias),
        np.array(ours_bias),
    )


def render():
    st.header("Demo 6 — The Distribution-Shift Trap")
    st.markdown(
        """
        In practice, your **calibration set** may come from a different distribution
        than your **test set** — different topics, time periods, or model versions.
        Most bias-correction methods break under this shift. Only the Rogan-Gladen
        estimator used here stays unbiased.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        q0         = st.slider("Judge specificity q0", 0.51, 1.0, 0.70, 0.01, key="ds_q0")
        q1         = st.slider("Judge sensitivity q1", 0.51, 1.0, 0.90, 0.01, key="ds_q1")
    with col2:
        theta_test = st.slider("True accuracy on test set θ_test", 0.1, 0.9, 0.65, 0.01, key="ds_theta")

    if q0 + q1 <= 1.0:
        st.error("q0 + q1 must be > 1.")
        return

    calib_prevs = np.linspace(0.05, 0.95, 200)
    naive_b, ppi_b, calib_b, ours_b = _compute_biases(theta_test, q0, q1, calib_prevs)

    theta_star = bias_crossover_threshold(q0, q1)

    fig = go.Figure()

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5,
                  annotation_text="Unbiased", annotation_position="right")

    fig.add_trace(go.Scatter(
        x=calib_prevs, y=naive_b,
        mode="lines", name="Naive (p̂)",
        line=dict(color="#EF553B", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=calib_prevs, y=ppi_b,
        mode="lines", name="PPI / Difference",
        line=dict(color="#AB63FA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=calib_prevs, y=calib_b,
        mode="lines", name="Calibration-only",
        line=dict(color="#FFA15A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=calib_prevs, y=ours_b,
        mode="lines", name="Ours (bias-corrected)",
        line=dict(color="#00CC96", width=3),
    ))

    # Mark where calib_prev == theta_test (no shift)
    fig.add_vline(
        x=theta_test,
        line_dash="dot", line_color="gray",
        annotation_text=f"No shift\n(θ_calib = θ_test = {theta_test:.2f})",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Calibration set prevalence θ_calib",
        yaxis_title="Estimator bias (estimate − truth)",
        xaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why do the other methods break?"):
        p_test = expected_naive_score(theta_test, q0, q1)
        st.markdown(f"""
**Naive**: Always biased by `E[p̂] − θ = {p_test - theta_test:+.3f}` (independent of calibration — it ignores calibration entirely).

**PPI / Difference estimator**: `p_test − p_calib + θ_calib`. When θ_calib ≠ θ_test, the
"p_calib" term doesn't cancel the bias in p_test. It only works when the two distributions match.

**Calibration-only**: Uses θ_calib directly as the estimate of θ_test. This equals the test θ only
when calibration and test distributions are identical — a strong assumption.

**Ours**: The correction `θ̂ = (p̂ + q0 − 1)/(q0 + q1 − 1)` depends only on the **confusion
matrix** P(Ẑ | Z), which is a property of the judge — not the prevalence P(Z).
As long as the judge behaves the same way on both sets, the estimator is unbiased
regardless of how different the prevalences are.
        """)
