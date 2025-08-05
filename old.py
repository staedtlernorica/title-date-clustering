import pandas as pd
import hdbscan
import re
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np

# === Configuration ===
# INPUT_CSV = "extra history all.csv"
INPUT_CSV = "eh.csv"
OUTPUT_CSV = "extra_history_series_grouped.csv"
SUMMARY_FILE = "series_summary.txt"
USE_TIME_CLUSTERING = False         # Toggle this to enable/disable time-based clustering
MAX_GAP_DAYS = 22                 # Max gap for episodes in a series if time clustering is used
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 2  # Minimum size for HDBSCAN clusters


# === Helpers ===
def clean_title(title):
    title = title.strip()
    # title = re.sub(r'\s*\|\s*', ' - ', title)
    # title = re.sub(r'\s*-\s*Extra History', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'\s*-\s*European History', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'\s*-\s*World War II', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'\s*-\s*WW2', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'\s*-\s*LIES', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'\s*-\s*Part\s*\d+', '', title, flags=re.IGNORECASE)
    # title = re.sub(r'(.*?)(\d+\s*:\s*)', r'\1', title)
    return title.strip()

def extract_common_series_name(titles):
    cleaned_titles = [clean_title(t) for t in titles]
    tokenized = [t.split(' ') for t in cleaned_titles]

    print(tokenized)

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

def run_time_based_clustering(df, max_gap_days=22):
    df["final_cluster"] = -1
    final_series_names = {}
    final_cluster_id = 0

    for t_cluster in df["title_cluster"].unique():
        if t_cluster == -1:
            continue
        group = df[df["title_cluster"] == t_cluster].copy()

        # DBSCAN based on days since start
        dates = group["days_since_start"].values.reshape(-1, 1)
        dbscan = DBSCAN(eps=max_gap_days, min_samples=2, metric='euclidean')
        group["date_subcluster"] = dbscan.fit_predict(dates)

        for sub_id in group["date_subcluster"].unique():
            if sub_id == -1:
                continue
            sub = group[group["date_subcluster"] == sub_id]
            if len(sub) < 2:
                continue

            df.loc[sub.index, "final_cluster"] = final_cluster_id
            name = extract_common_series_name(sub["title"].tolist())
            final_series_names[final_cluster_id] = name
            final_cluster_id += 1

    return df, final_series_names

def assign_final_clusters_from_title(df):
    df["final_cluster"] = -1
    final_series_names = {}
    for cluster_id in sorted(df["title_cluster"].unique()):
        if cluster_id == -1:
            continue
        sub = df[df["title_cluster"] == cluster_id]
        if len(sub) < 2:
            continue
        df.loc[sub.index, "final_cluster"] = cluster_id
        name = extract_common_series_name(sub["title"].tolist())
        final_series_names[cluster_id] = name
    return df, final_series_names


# === Main Pipeline ===
def main():
    # Step 1: Load CSV
    df = pd.read_csv(INPUT_CSV)
    # df.columns = [
    #     "title", "published", "views", "likes", "comments",
    #     "video_id", "duration", "url"
    # ]
    df["published"] = pd.to_datetime(df["published"])
    df["days_since_start"] = (df["published"] - df["published"].min()).dt.days

    

    # Step 2: Embed titles
    print("🔍 Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["title"].tolist())

    # Step 3: Initial HDBSCAN clustering
    print("📊 Running HDBSCAN on titles...")
    title_clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    df["title_cluster"] = title_clusterer.fit_predict(embeddings)

    # Step 4 & 5: Apply either time-based or title-only clustering
    print(f"🧩 Using {'time-based' if USE_TIME_CLUSTERING else 'title-only'} clustering...")
    if USE_TIME_CLUSTERING:
        df, final_series_names = run_time_based_clustering(df, MAX_GAP_DAYS)
    else:
        df, final_series_names = assign_final_clusters_from_title(df)

    # Step 6: Assign series names
    df["series_name"] = df["final_cluster"].apply(lambda cid: final_series_names.get(cid, "Unclustered"))

    # Step 7: Sort & save results
    df = df.sort_values(by=["series_name", "published"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Output saved to {OUTPUT_CSV}")

    # Step 8: Write summary file
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("📚 Series Groups with Titles:\n")
        
        # Write clustered entries
        for sid, name in final_series_names.items():
            cluster_df = df[df["final_cluster"] == sid].sort_values(by="published")
            f.write(f"\n[{sid}] {name} — {len(cluster_df)} videos\n")
            for _, row in cluster_df.iterrows():
                f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")

        # Write unclustered entries
        unclustered_df = df[df["final_cluster"] == -1].sort_values(by="published")
        if not unclustered_df.empty:
            f.write(f"\n[-1] Unclustered — {len(unclustered_df)} videos\n")
            for _, row in unclustered_df.iterrows():
                f.write(f"  - {row['published'].date()} | {row['title']} | {row['url']}\n")


if __name__ == "__main__":
    main()
