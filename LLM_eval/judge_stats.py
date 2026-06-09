"""
Statistical core for LLM-as-a-judge bias correction.
All estimators, CIs, and derived quantities live here.
Reference: Lee, Zeng, Jeong, Sohn, Lee (arXiv:2511.21140), Appendix C.

Notation:
  theta  : true accuracy
  p      : naive judge score (fraction judged correct)
  q1     : sensitivity  = P(Z_hat=1 | Z=1)
  q0     : specificity  = P(Z_hat=0 | Z=0)
  n      : test-set size
  m0, m1 : calibration examples with true label 0 / 1
"""

from math import sqrt
import numpy as np
from scipy.stats import norm


def _clip(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def _check_judge(q0, q1):
    if q0 + q1 <= 1.0:
        raise ValueError(
            f"Judge must satisfy q0 + q1 > 1 (got q0={q0:.3f}, q1={q1:.3f}). "
            "A judge worse than random chance cannot be corrected."
        )


# ---------------------------------------------------------------------------
# Core estimators (paper Appendix C — keep exactly)
# ---------------------------------------------------------------------------

def point_estimator(p, q0, q1):
    """Bias-corrected point estimate of true accuracy theta."""
    _check_judge(q0, q1)
    th = (p + q0 - 1) / (q0 + q1 - 1)
    return _clip(th)


def confidence_interval(p, q0, q1, n, m0, m1, alpha=0.05):
    """Adjusted (1 - alpha) confidence interval for theta."""
    _check_judge(q0, q1)
    z = norm.ppf(1 - alpha / 2)
    p  = (n * p  + z**2 / 2) / (n + z**2)
    q0 = (m0 * q0 + 1) / (m0 + 2)
    q1 = (m1 * q1 + 1) / (m1 + 2)
    n, m0, m1 = n + z**2, m0 + 2, m1 + 2
    th = (p + q0 - 1) / (q0 + q1 - 1)
    dth = 2 * z**2 * (-(1 - th) * q0 * (1 - q0) / m0 + th * q1 * (1 - q1) / m1)
    se = sqrt(p * (1 - p) / n + (1 - th)**2 * q0 * (1 - q0) / m0
              + th**2 * q1 * (1 - q1) / m1) / (q0 + q1 - 1)
    return _clip(th + dth - z * se), _clip(th + dth + z * se)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def bias_crossover_threshold(q0, q1):
    """theta* where naive score switches from over- to under-estimation."""
    _check_judge(q0, q1)
    return (1 - q0) / (2 - q0 - q1)


def expected_naive_score(theta, q0, q1):
    """E[p_hat] as a function of true accuracy theta."""
    return (q0 + q1 - 1) * theta + (1 - q0)


def naive_ci(p, n, alpha=0.05):
    """Naive 95% CI for p (NOT for theta) — the cautionary baseline."""
    z = norm.ppf(1 - alpha / 2)
    se = sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


# ---------------------------------------------------------------------------
# Asymptotic variance (n -> inf limit used for budget calculations)
# ---------------------------------------------------------------------------

def asymptotic_variance(theta_hat, p, q0, q1, n, m0, m1):
    """Full asymptotic variance of the bias-corrected estimator."""
    _check_judge(q0, q1)
    denom = (q0 + q1 - 1) ** 2
    return (
        p * (1 - p) / n
        + (1 - theta_hat) ** 2 * q0 * (1 - q0) / m0
        + theta_hat ** 2 * q1 * (1 - q1) / m1
    ) / denom


def ci_length_equal_split(p, q0, q1, m, alpha=0.05):
    """CI half-width for symmetric calibration split m0 = m1 = m/2, n -> inf."""
    _check_judge(q0, q1)
    z = norm.ppf(1 - alpha / 2)
    th = _clip((p + q0 - 1) / (q0 + q1 - 1))
    m0 = m1 = m / 2
    se = sqrt(
        (1 - th) ** 2 * q0 * (1 - q0) / m0 + th ** 2 * q1 * (1 - q1) / m1
    ) / (q0 + q1 - 1)
    return 2 * z * se


def adaptive_allocation(p, q0, q1, m_total, alpha=0.05):
    """
    Optimal m0, m1 under fixed total budget m0+m1 = m_total (n -> inf).
    Returns (m0, m1, ci_length).
    """
    _check_judge(q0, q1)
    z = norm.ppf(1 - alpha / 2)
    th = _clip((p + q0 - 1) / (q0 + q1 - 1))
    kappa = (1 - q0) / max(1 - q1, 1e-9)
    # m0/m1 = (1/p - 1) * sqrt(kappa)  ... from paper
    ratio = ((1 / max(p, 1e-9)) - 1) * sqrt(kappa)
    m1 = m_total / (1 + ratio)
    m0 = m_total - m1
    m0, m1 = max(m0, 1), max(m1, 1)
    se = sqrt(
        (1 - th) ** 2 * q0 * (1 - q0) / m0 + th ** 2 * q1 * (1 - q1) / m1
    ) / (q0 + q1 - 1)
    return m0, m1, 2 * z * se


def min_calibration_for_width(p, q0, q1, target_width, alpha=0.05,
                               adaptive=False, m_max=10_000):
    """Binary search for smallest m achieving target CI full-width."""
    _check_judge(q0, q1)
    for m in range(2, m_max + 1, 2):
        if adaptive:
            _, _, length = adaptive_allocation(p, q0, q1, m, alpha)
        else:
            length = ci_length_equal_split(p, q0, q1, m, alpha)
        if length <= target_width:
            return m
    return m_max


# ---------------------------------------------------------------------------
# LLM-vs-human region (symmetric judge: q0 = q1 = q)
# ---------------------------------------------------------------------------

def llm_beats_human_region(q):
    """
    For symmetric judge quality q, return (theta_lo, theta_hi) where
    calibrated LLM beats human-only at equal total budget (n->inf).
    Returns None if q <= 0.5 + 1/(2*sqrt(2)) ~ 0.854.
    """
    threshold = 0.5 + 1 / (2 * sqrt(2))
    if q <= threshold:
        return None
    half_width = sqrt(0.5 - 1 / (4 * (2 * q - 1) ** 2))
    return max(0.0, 0.5 - half_width), min(1.0, 0.5 + half_width)


# ---------------------------------------------------------------------------
# Distribution-shift estimators (Section 8 comparisons)
# ---------------------------------------------------------------------------

def naive_estimator_bias(theta_test, q0, q1, prev_calib):
    """Bias of the naive score p_hat as a function of test theta."""
    return expected_naive_score(theta_test, q0, q1) - theta_test


def ppi_estimator(p_test, p_calib, theta_calib, q0, q1):
    """Prediction-powered inference (difference) estimator."""
    return p_test - p_calib + theta_calib


def calibration_only_estimator(theta_calib):
    """Calibration-set-only estimator (ignores test LLM scores)."""
    return theta_calib


def our_estimator_bias_under_shift(theta_test, q0, q1):
    """Our estimator stays unbiased regardless of calibration prevalence."""
    return 0.0
