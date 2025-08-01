import pandas as pd
import hdbscan
import re
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np

# === Step 1: Load CSV ===
df = pd.read_csv("all extra history.csv", header=None)
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
]
df["published"] = pd.to_datetime(df["published"])
df["days_since_start"] = (df["published"] - df["published"].min()).dt.days

# === Step 2: Create title embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["title"].tolist())

# === Step 3: Initial clustering by title ===
title_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
df["title_cluster"] = title_clusterer.fit_predict(embeddings)

# === Step 4: Clean titles ===
def clean_title(title):
    title = title.strip()
    title = re.sub(r'\s*\|\s*', ' - ', title)
    title = re.sub(r'\s*-\s*Extra History', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*European History', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*World War II', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*WW2', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*LIES', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*Part\s*\d+', '', title, flags=re.IGNORECASE)
    title = re.sub(r'(.*?)(\d+\s*:\s*)', r'\1', title)
    return title.strip()

def extract_common_series_name(titles):
    cleaned_titles = [clean_title(t) for t in titles]
    tokenized = [t.split(' - ') for t in cleaned_titles]

    ngram_counter = Counter()
    for tokens in tokenized:
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens) + 1):
                chunk = ' - '.join(tokens[i:j])
                ngram_counter[chunk] += 1

    min_count = max(2, len(titles) // 2)
    common_ngrams = [ng for ng, count in ngram_counter.items() if count >= min_count]
    common_ngrams.sort(key=lambda x: (-len(x), -ngram_counter[x]))

    return common_ngrams[0] if common_ngrams else "Unnamed Series"

# === Step 5: Date-based pruning within title clusters using DBSCAN ===
df["final_cluster"] = -1
final_series_names = {}
final_cluster_id = 0

# Max allowed days between episodes in a series
max_gap_days = 22

for t_cluster in df["title_cluster"].unique():
    if t_cluster == -1:
        continue
    group = df[df["title_cluster"] == t_cluster].copy()

    # Use raw days for DBSCAN (no scaling)
    dates = group["days_since_start"].values.reshape(-1, 1)
    dbscan = DBSCAN(eps=max_gap_days, min_samples=2, metric='euclidean')
    group["date_subcluster"] = dbscan.fit_predict(dates)

    for sub_id in group["date_subcluster"].unique():
        if sub_id == -1:
            continue
        sub = group[group["date_subcluster"] == sub_id]
        if len(sub) < 2:
            continue

        # Assign final cluster ID and series name
        df.loc[sub.index, "final_cluster"] = final_cluster_id
        name = extract_common_series_name(sub["title"].tolist())
        final_series_names[final_cluster_id] = name
        final_cluster_id += 1

# Add series name to dataframe
df["series_name"] = df["final_cluster"].apply(lambda cid: final_series_names.get(cid, "Unclustered"))

# === Step 6: Sort & Save ===
df = df.sort_values(by=["series_name", "published"])
df.to_csv("extra_history_series_grouped.csv", index=False)

# === Step 7: Save Summary to File ===
output_path = "series_summary.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("📚 Series Groups with Titles:\n")
    for sid, name in final_series_names.items():
        cluster_df = df[df["final_cluster"] == sid].sort_values(by="published")
        f.write(f"\n[{sid}] {name} — {len(cluster_df)} videos\n")
        for _, row in cluster_df.iterrows():
            f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")

print(f"✅ Series summary written to {output_path}")
