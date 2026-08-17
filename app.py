"""
app.py

Streamlit web UI over the same classify.py pipeline run.py uses -
lets someone try the tool in a browser without cloning the repo or
using a terminal. Same underlying logic either way; this is a thin
presentation layer, not a second implementation.

    streamlit run app.py
"""

import streamlit as st

from classify import CONFIDENCE_GAP_THRESHOLD, build_index, classify, confidence_gap
from duty import fetch_standard_duty_rate
from references import DEFAULT_NOMENCLATURE_PATH, load_references

st.set_page_config(page_title="HS Code Classifier", page_icon="📦")


@st.cache_resource(show_spinner="Loading reference data and building index (first run only)...")
def _load_index():
    search_refs = load_references()
    nomenclature_refs = load_references(DEFAULT_NOMENCLATURE_PATH)
    return build_index(search_refs + nomenclature_refs), len(search_refs), len(nomenclature_refs)


def _format_code(code: str) -> str:
    return f"{code[:4]} {code[4:6]} {code[6:8]} {code[8:10]}"


index, n_search_refs, n_nomenclature = _load_index()

st.title("HS Code Classifier")
st.caption(
    f"Retrieval over {n_search_refs:,} UK Trade Tariff reference titles + "
    f"{n_nomenclature:,} official nomenclature entries. Not legal advice - "
    "narrows candidates for a human classifier to verify, see README."
)

query = st.text_input("Describe the product", placeholder="e.g. wireless bluetooth headphones")
use_rerank = st.checkbox("Use cross-encoder re-ranking", value=True)

if query:
    with st.spinner("Classifying..."):
        candidates = classify(query, index, top_k=5, use_rerank=use_rerank)

    if not candidates:
        st.warning("No matches found.")
    else:
        gap = confidence_gap(candidates)
        if gap is not None and gap < CONFIDENCE_GAP_THRESHOLD:
            st.warning(
                "Low confidence: the top candidates scored close together - this description "
                "may be genuinely ambiguous. Treat these as a starting point, not a settled answer."
            )

        for i, c in enumerate(candidates, start=1):
            with st.container(border=True):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{_format_code(c.code)}**")
                cols[0].caption(f"matched: {c.title!r} ({c.level})")
                cols[1].metric("score", f"{c.score:.2f}")

                if i == 1:
                    duty_rate = fetch_standard_duty_rate(c.code)
                    if duty_rate:
                        st.caption(f"Standard (non-preferential) duty rate: {duty_rate}")
