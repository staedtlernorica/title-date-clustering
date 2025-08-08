import pandas as pd
import hdbscan
import re
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np

# === Step 1: Load CSV ===
df = pd.read_csv("eh no shorts musics.csv", header=None)
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
]
df["published"] = pd.to_datetime(df["published"])
df["days_since_start"] = (df["published"] - df["published"].min()).dt.days

# === Step 2: Create title embeddings ===
print("🔄 Generating title embeddings...")
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(df["title"].tolist())

# === Step 3: Initial clustering by title ===
print("🔍 Running HDBSCAN title clustering...")
title_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
df["title_cluster"] = title_clusterer.fit_predict(embeddings)

print("\n=== Title Clustering Results ===")
for cluster_id in sorted(df["title_cluster"].unique()):
    cluster_df = df[df["title_cluster"] == cluster_id]
    print(f"\nCluster {cluster_id} — {len(cluster_df)} videos")
    for title in cluster_df["title"]:
        print(f"  - {title}")

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
    print("\n🧠 Extracting common series name from:")
    for t in titles:
        print(f"  - {t}")
    
    cleaned_titles = [clean_title(t) for t in titles]
    print("Cleaned Titles:")
    for ct in cleaned_titles:
        print(f"  - {ct}")
    
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

max_gap_days = 22

for t_cluster in df["title_cluster"].unique():
    if t_cluster == -1:
        continue
    group = df[df["title_cluster"] == t_cluster].copy()

    dates = group["days_since_start"].values.reshape(-1, 1)
    dbscan = DBSCAN(eps=max_gap_days, min_samples=2, metric='euclidean')
    group["date_subcluster"] = dbscan.fit_predict(dates)

    print(f"\n--- 📆 Date Subclustering for Title Cluster {t_cluster} ---")
    for sub_id in group["date_subcluster"].unique():
        sub = group[group["date_subcluster"] == sub_id]
        print(f"\n  Subcluster {sub_id} — {len(sub)} videos")
        for title, date in zip(sub["title"], sub["published"]):
            print(f"    - {date.date()} | {title}")

        if sub_id == -1 or len(sub) < 2:
            continue

        df.loc[sub.index, "final_cluster"] = final_cluster_id
        name = extract_common_series_name(sub["title"].tolist())
        final_series_names[final_cluster_id] = name
        final_cluster_id += 1

# Add series name to dataframe
df["series_name"] = df["final_cluster"].apply(lambda cid: final_series_names.get(cid, "Unclustered"))

# === Step 6: Sort & Save CSV ===
df = df.sort_values(by=["series_name", "published"])
df.to_csv("extra_history_series_grouped.csv", index=False)

# === Step 7: Save Summary to File ===
output_path = "series_summary.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("📚 Series Groups with Titles:\n")

    # Write clustered entries
    for sid, name in final_series_names.items():
        cluster_df = df[df["final_cluster"] == sid].sort_values(by="published")
        f.write(f"\n[{sid}] {name} — {len(cluster_df)} videos\n")
        for _, row in cluster_df.iterrows():
            f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")

    # Write unclustered entries
    unclustered_df = df[df["final_cluster"] == -1].sort_values(by="published")
    f.write(f"\n[Unclustered] — {len(unclustered_df)} videos\n")
    for _, row in unclustered_df.iterrows():
        f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")

print(f"✅ Series summary written to {output_path}")
