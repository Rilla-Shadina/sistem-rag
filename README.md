**Sistem Question Answering Retrieval-Augmented Generation (RAG)**

Proyek ini membangun sistem Retrieval-Augmented Generation (RAG) yang mampu menjawab 
pertanyaan pengguna berdasarkan kumpulan artikel berita. Sistem dirancang agar jawaban 
yang dihasilkan hanya bersumber dari dokumen relevan

**Dataset:**
- News Category dataset (Kaggle)
- Berjumlah ±100.000 artikel
- Fokus domain: Artikel Kesehatan
- Berbahasa Inggris

**Fitur Utama:**
- Pencarian dokumen menggunakan semantic search berbasis embedding
- Ekspansi kata kunci dan pemrosesan query kesehatan
- Ekstraksi konteks berbasis kalimat
- Jawaban (generation) berbasis instruksi menggunakan FLAN-T5
- Mekanisme fallback jika jawaban tidak ditemukan

**Teknologi yang Digunakan:**
Python, HuggingFace Transformers, FLAN-T5, BM25 (rank-bm25), Pandas, Regular Expression, Streamlit (dashboard).

**Kompetensi yang Ditunjukkan:**
- Natural Language Processing (NLP)
- Information Retrieval (IR)
- Data Cleaning dan Preprocessing Teks
- Perancangan Sistem Modular

**Hasil Proyek:**
Sistem dapat mengidentifikasi topik dari dataset berita kesehatan dan menghasilkan jawaban yang relevan 
dan ringkas berdasarkan informasi yang ada di dokumen.
