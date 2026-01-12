# preprocessing_revised.py
import json
import re
import os
import pandas as pd

PATH = "News_Category_Dataset_v3.json"
os.makedirs("output", exist_ok=True)

# =====================
# Load JSON
# =====================
with open(PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# =====================
# Gabungkan teks (REVISI)
# =====================
df["text"] = (
    df["headline"].fillna("") + ". " +
    df["short_description"].fillna("") + ". " +
    "Category: " + df["category"].fillna("")
)

# =====================
# Cleaning
# =====================
def clean_text(t):
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s\.\!\?]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

df["text_clean"] = df["text"].apply(clean_text)

# =====================
# Simpan TANPA chunking
# =====================
df_out = df[["headline", "category", "text_clean"]].copy()
df_out.rename(columns={"text_clean": "text"}, inplace=True)
df_out.to_pickle("output/news_clean.pkl")

print("Saved output/news_clean.pkl")
