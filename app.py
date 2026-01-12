import streamlit as st
from retrieve import retrieve_docs
from rag_generator import generate_answer

st.set_page_config(
    page_title="Sistem RAG – News Article QA",
    layout="centered"
)

st.title("📰 Sistem RAG – News Article QA")
st.write("Jawaban dihasilkan **hanya berdasarkan dokumen berita**.")

query = st.text_input(
    "Masukkan pertanyaan:",
    placeholder="contoh: health effects of sleep deprivation"
)

TOP_K = 5

if st.button("Cari Jawaban"):
    if not query.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("🔍 Mencari dokumen relevan..."):
            docs = retrieve_docs(query, top_k=TOP_K)

        st.subheader("🔍 Dokumen Relevan")
        for i, d in enumerate(docs, 1):
            with st.expander(f"{i}. {d['headline']} ({d['category']})"):
                st.write(d["text"])

        with st.spinner("🤖 Menghasilkan jawaban..."):
            answer = generate_answer(query, docs)

        st.subheader("🤖 Jawaban")
        st.write(answer)
