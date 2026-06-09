"""
Streamlit app — LLM-as-a-Judge: How to Report It Correctly
One page per demo; navigate via the sidebar.
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="LLM Judge Evaluations — Interactive Demos",
    page_icon="⚖️",
    layout="wide",
)

from demos import leaderboard, bias_curve, coverage, budget, llm_vs_human, distshift

DEMOS = {
    "1 · Leaderboard That Lies":      leaderboard,
    "2 · Bias Curve":                 bias_curve,
    "3 · Coverage Gut-Punch":         coverage,
    "4 · Budget Calculator":          budget,
    "5 · LLM vs Human Map":           llm_vs_human,
    "6 · Distribution-Shift Breaker": distshift,
}

st.sidebar.title("⚖️ LLM Judge Demos")
st.sidebar.markdown(
    "Interactive companion to the paper  \n"
    "*How to Correctly Report LLM-as-a-Judge Evaluations*  \n"
    "(Lee, Zeng, Jeong, Sohn, Lee — arXiv:2511.21140)"
)
st.sidebar.markdown("---")

choice = st.sidebar.radio("Select demo", list(DEMOS.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Key notation**\n"
    "- θ : true accuracy\n"
    "- p̂ : naive judge score\n"
    "- q0 : judge specificity\n"
    "- q1 : judge sensitivity\n"
    "- m0/m1 : calibration labels (wrong/correct)\n"
    "- θ\\* : bias crossover threshold"
)

DEMOS[choice].render()
