import pickle
import re

data = pickle.load(open("output/bm25_index.pkl", "rb"))
bm25 = data["bm25"]
df = data["df"]

def clean_query(q):
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()

def retrieve_docs(query, top_k=5):
    tokens = clean_query(query).split()
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        results.append({
            "text": df.iloc[i]["text"],
            "headline": df.iloc[i]["headline"],
            "category": df.iloc[i]["category"]
        })

    return results

if __name__ == "__main__":
    q = "health and wellness"
    docs = retrieve_docs(q)
    for d in docs:
        print("----")
        print(d["text"][:200])
