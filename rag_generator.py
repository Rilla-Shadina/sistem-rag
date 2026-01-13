from transformers import pipeline
import re

# =====================
# Load model
# =====================
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# =====================
# Constants
# =====================
STOPWORDS = {
    "what", "is", "are", "the", "of", "and", "to", "in", "on", "for", "with"
}

KEYWORD_MAP = {
    "effects": {"effects", "impact", "issues", "problems", "risks"},
    "health": {"health", "wellbeing", "well-being"},
    "wellness": {"wellness", "health", "lifestyle"},
    "sleep": {"sleep", "sleeping"},
    "tv": {"tv", "television"},
    "mental": {"mental", "psychological"}
}

# =====================
# Utilities
# =====================
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

def tokenize(text):
    return [
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]

def expand_query_terms(query):
    words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
    expanded = set()

    for w in words:
        if w in STOPWORDS:
            continue
        if w in KEYWORD_MAP:
            expanded |= KEYWORD_MAP[w]
        else:
            expanded.add(w)

    return expanded

# =====================
# Intent Detection
# =====================
def is_topic_question(query):
    q = query.lower()
    return "topic" in q or "topics" in q

# =====================
# Document Filtering
# =====================
def filter_health_docs(docs):
    allowed = {"WELLNESS", "WOMEN"}
    return [d for d in docs if d.get("category") in allowed]

# =====================
# Topic Extraction (TITLE-BASED)
# =====================
def extract_topics_from_titles(docs):
    topics = set()

    for d in docs:
        title = d.get("title", "").lower()
        title = re.sub(r"[^a-z0-9\s]", " ", title)

        if "mental" in title and "health" in title:
            if "athlete" in title:
                topics.add("mental health of athletes")
            else:
                topics.add("mental health")

        if "women" in title and "health" in title:
            topics.add("women’s health")

    return topics


# =====================
# Sentence Retrieval 
# =====================
def get_relevant_sentences(query, docs, max_per_doc=2):
    q_words = expand_query_terms(query)
    selected = []

    for d in docs:
        sentences = extract_sentences(d.get("text", ""))
        scored = []

        for s in sentences:
            s_tokens = set(tokenize(s))
            overlap = q_words & s_tokens
            score = len(overlap)

            if score >= 1:
                scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected.extend([s for _, s in scored[:max_per_doc]])

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
# Generation
# =====================
def generate_answer(query, docs, max_length=160):

    if not docs:
        return "Informasi tidak ditemukan."

    # =====================
    # HARD TOPIC EXTRACTION (ANTI-GAGAL)
    # =====================
    topics = set()

    for d in docs:
        text = clean_text(d.get("text", ""))

        if not text:
            continue

        if "mental" in text and "health" in text:
            topics.add("mental health")

        if "athlete" in text and "health" in text:
            topics.add("mental health of athletes")

        if "women" in text and "health" in text:
            topics.add("women’s health")

        if "sleep" in text:
            topics.add("sleep health")

        if "tv" in text or "television" in text:
            topics.add("health effects of television")

    if topics:
        return (
            "The articles discuss the following health and wellness topics: "
            + ", ".join(sorted(topics)) + "."
        )

    # ==============
    # FALLBACK QA 
    # ==============
    relevant = get_relevant_sentences(query, docs)
    if not relevant:
        return "Informasi tidak ditemukan."

    context = build_context(relevant)

    prompt = f"""
Answer the question using only the information below.

DOCUMENTS:
{context}

QUESTION:
{query}

ANSWER:
"""

    out = generator(prompt, max_length=max_length, do_sample=False)
    return out[0]["generated_text"].strip()
