import pickle
import re
import pandas as pd
from rank_bm25 import BM25Okapi

# =====================
# Load cleaned data
# =====================
df = pd.read_pickle("output/news_clean.pkl")

# =====================
# Tokenization (HARUS KONSISTEN DENGAN QUERY)
# =====================
STOPWORDS = {
    "the", "is", "are", "of", "and", "to", "in", "on", "for", "with",
    "a", "an", "that", "this", "it", "as", "by", "from"
}

def bm25_tokenize(text: str):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

# =====================
# Build BM25 index
# =====================
tokenized_docs = [bm25_tokenize(text) for text in df["text"]]
bm25 = BM25Okapi(tokenized_docs)

# =====================
# Save index + metadata
# =====================
with open("output/bm25_index.pkl", "wb") as f:
    pickle.dump(
        {
            "bm25": bm25,
            "df": df
        },
        f
    )

print("✅ BM25 index successfully saved at output/bm25_index.pkl")
print(f"📄 Total documents indexed: {len(tokenized_docs)}")
