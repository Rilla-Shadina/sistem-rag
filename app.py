import streamlit as st
from retrieve import retrieve_docs
from rag_generator import generate_health_answer

# =====================
# Konfigurasi halaman
# =====================
st.set_page_config(
    page_title="Sistem RAG – News Article QA",
    layout="centered"
)

st.title("📰 Sistem RAG – News Article QA")
st.write("Jawaban dihasilkan **hanya berdasarkan dokumen berita kesehatan yang relevan**.")

# =====================
# Input query
# =====================
query = st.text_input(
    "Masukkan pertanyaan:",
    placeholder="contoh: health effects of sleep deprivation"
)

TOP_K = 5  # jumlah dokumen/kalimat yang diambil

# =====================
# Tombol Cari Jawaban
# =====================
if st.button("Cari Jawaban"):
    if not query.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        # Ambil dokumen relevan
        with st.spinner("Mencari dokumen relevan..."):
            docs = retrieve_docs(query, top_k=TOP_K)

        if not docs:
            st.warning("Tidak ditemukan dokumen yang relevan.")
        else:
            # Tampilkan dokumen relevan
            st.subheader("🔍 Dokumen Relevan")
            for i, d in enumerate(docs, 1):
                with st.expander(f"{i}. {d.get('headline', 'Tidak ada judul')} ({d.get('category', 'Tidak ada kategori')})"):
                    st.write(d.get("text", "Tidak ada isi dokumen."))

            # Hasilkan jawaban
            with st.spinner("Menghasilkan jawaban..."):
                answer = generate_health_answer(query, docs)

            st.subheader("Jawaban")
            st.write(answer)
