"""Streamlit dashboard for the Hybrid RAG API."""

from __future__ import annotations

from pathlib import Path

import httpx
import streamlit as st

API_URL = "http://api:8000"

st.set_page_config(page_title="Hybrid RAG Pipeline", layout="wide")
st.title("Hybrid RAG Pipeline")

tab_query, tab_eval = st.tabs(["Ask", "Evaluation"])

with tab_query:
    dense_only = st.toggle("Dense only", value=False)
    rrf_weight = st.slider("Dense weight", 0.0, 1.0, 0.7, 0.05, disabled=dense_only)
    query = st.text_input("Question", placeholder="How does the ingestion pipeline handle unsupported files?")
    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving and generating..."):
            response = httpx.post(
                f"{API_URL}/v1/ask",
                json={"query": query, "dense_only": dense_only, "rrf_weight": rrf_weight},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

        if data["confidence_score"] < 0.4:
            st.warning(f"Low confidence: {data['confidence_score']:.2f}")
        else:
            st.metric("Confidence", f"{data['confidence_score']:.2f}")
        st.markdown(data["answer"])

        st.subheader("Citations")
        for citation in data["citations"]:
            with st.expander(f"[{citation['index']}] {citation['source']} - {citation['verification_status']}"):
                st.write(citation["chunk_text"])

        st.subheader("Retrieval Trace")
        st.dataframe(data["retrieved_chunks"], use_container_width=True)
        st.caption(f"Latency: {data['latency_ms']}")

with tab_eval:
    report = Path("eval/report.md")
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
        chart = Path("eval/hybrid_vs_dense.png")
        if chart.exists():
            st.image(str(chart))
    else:
        st.info("Run the evaluation report generator to populate eval/report.md.")
