import pandas as pd
import hdbscan
import re
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np

# === Load Data ===
df = pd.read_csv("eh no shorts musics.csv", header=None)
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
]
df["published"] = pd.to_datetime(df["published"])
df["days_since_start"] = (df["published"] - df["published"].min()).dt.days
df["final_cluster"] = -1  # Initialize

# === Sentence Embedding Model ===
model = SentenceTransformer("all-mpnet-base-v2")

# === Title Cleaning ===
def clean_title(title):
    title = title.strip()
    title = re.sub(r'\s*\|\s*', ' - ', title)
    title = re.sub(r'\s*-\s*(Extra History|European History|World War II|WW2|LIES)', '', title, flags=re.IGNORECASE)
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

# === Clustering Function ===
def run_clustering(df_subset, start_cluster_id=0, pass_id=1):
    cluster_id = start_cluster_id
    series_names = {}

    print(f"\n🔁 Clustering Pass {pass_id}: {len(df_subset)} videos")

    # Step 1: Title Embeddings & HDBSCAN
    embeddings = model.encode(df_subset["title"].tolist())
    title_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
    df_subset["title_cluster"] = title_clusterer.fit_predict(embeddings)

    # Step 2: Date-Based Subclustering with DBSCAN
    for t_cluster in df_subset["title_cluster"].unique():
        if t_cluster == -1:
            continue
        group = df_subset[df_subset["title_cluster"] == t_cluster].copy()
        dates = group["days_since_start"].values.reshape(-1, 1)

        dbscan = DBSCAN(eps=22, min_samples=2)
        group["date_subcluster"] = dbscan.fit_predict(dates)

        for sub_id in group["date_subcluster"].unique():
            sub = group[group["date_subcluster"] == sub_id]
            if sub_id == -1 or len(sub) < 2:
                continue

            df_subset.loc[sub.index, "final_cluster"] = cluster_id
            name = extract_common_series_name(sub["title"].tolist())
            series_names[cluster_id] = name
            cluster_id += 1

    return df_subset, series_names, cluster_id

# === First Clustering Pass ===
df_first, names_first, next_cluster_id = run_clustering(df.copy(), 0, pass_id=1)
df.update(df_first)
final_series_names = names_first

# === Second Clustering Pass (on unclustered) ===
unclustered = df[df["final_cluster"] == -1].copy()
if not unclustered.empty:
    df_second, names_second, _ = run_clustering(unclustered, next_cluster_id, pass_id=2)
    df.update(df_second)
    final_series_names.update(names_second)

# === Assign Series Names ===
df["series_name"] = df["final_cluster"].apply(lambda cid: final_series_names.get(cid, "Unclustered"))
df = df.sort_values(by=["series_name", "published"])

# === Save Results ===
df.to_csv("extra_history_series_grouped.csv", index=False)

with open("series_summary.txt", "w", encoding="utf-8") as f:
    f.write("📚 Series Groups with Titles:\n")

    for sid, name in final_series_names.items():
        cluster_df = df[df["final_cluster"] == sid].sort_values(by="published")
        f.write(f"\n[{sid}] {name} — {len(cluster_df)} videos\n")
        for _, row in cluster_df.iterrows():
            f.write(f"{row['title']} \ {row['url']}\n")

    unclustered_df = df[df["final_cluster"] == -1].sort_values(by="published")
    f.write(f"\n[Unclustered] — {len(unclustered_df)} videos\n")
    for _, row in unclustered_df.iterrows():
        # f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")
        f.write(f"{row['title']} \ {row['url']}\n")

print("✅ Clustering complete. Results saved to CSV and summary file.")
