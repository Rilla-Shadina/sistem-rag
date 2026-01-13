from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import re
import torch

# =====================
# Load models
# =====================
# FLAN-T5 base (lebih besar dari small, tetap bisa pakai CPU)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1  # -1 = CPU
)

# Embedding model ringan & cepat
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# =====================
# Utilities
# =====================
STOPWORDS = {"what", "is", "are", "the", "of", "and", "to", "in", "on", "for", "with"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_sentences(text, min_len=8):
    text = clean_text(text)
    parts = re.split(r"[.!?]", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]

# =====================
# Filter Health Docs
# =====================
def filter_health_docs(docs):
    health_categories = {"WELLNESS", "HEALTH", "WOMEN"}
    return [d for d in docs if d.get("category", "").upper() in health_categories]

# =====================
# Semantic Retrieval
# =====================
def get_relevant_sentences_semantic(query, docs, top_k=10, threshold=0.5):
    """
    Ambil kalimat paling relevan dengan query menggunakan embeddings.
    """
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    all_sentences = []

    for d in docs:
        text = d.get("text","")
        sentences = extract_sentences(text)
        all_sentences.extend(sentences)

    if not all_sentences:
        return []

    sentence_embs = embed_model.encode(all_sentences, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, sentence_embs)[0]
    sims = sims.cpu().numpy()

    # Pilih kalimat di atas threshold
    top_idx = [i for i, sim in enumerate(sims) if sim > threshold]
    if not top_idx:  # fallback jika tidak ada yang > threshold
        top_idx = sims.argsort()[::-1][:top_k]

    top_sentences = [all_sentences[i] for i in top_idx]
    return top_sentences

# =====================
# Fallback Keyword Search
# =====================
def get_relevant_sentences_keyword(query, docs):
    keywords = set(clean_text(query).split()) - STOPWORDS
    selected = []
    for d in docs:
        for s in extract_sentences(d.get("text","")):
            s_tokens = set(clean_text(s).split())
            if s_tokens & keywords:
                selected.append(s)
    return selected

# =====================
# Context Builder
# =====================
def build_context(sentences, max_sentences=6):
    seen = set()
    context = []

    for s in sentences:
        if s not in seen:
            context.append(s)
            seen.add(s)
        if len(context) >= max_sentences:
            break

    return "\n".join(f"- {s}" for s in context)

# =====================
# Generate Health Answer
# =====================
def generate_health_answer(query, docs, max_length=300, top_k=10, threshold=0.5):
    """
    Query terkait kesehatan -> jawaban menyimpulkan efek kesehatan
    """
    # Filter dokumen kesehatan
    health_docs = filter_health_docs(docs)
    if not health_docs:
        return "Informasi kesehatan tidak ditemukan di dataset."

    # Semantic search
    relevant_sentences = get_relevant_sentences_semantic(query, health_docs, top_k=top_k, threshold=threshold)

    # Fallback keyword search jika tidak ada semantic match
    if not relevant_sentences:
        relevant_sentences = get_relevant_sentences_keyword(query, health_docs)

    if not relevant_sentences:
        return "Informasi tidak ditemukan."

    context = build_context(relevant_sentences)

    prompt = f"""
    Read the following documents and answer the question below.
    Focus ONLY on the specific health effect mentioned.
    Do not include unrelated information.
    Write a concise answer in complete sentences.

    DOCUMENTS:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    out = generator(prompt, max_length=max_length, do_sample=False)
    return out[0]["generated_text"].strip()
