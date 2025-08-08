import pandas as pd

# Load CSV
df = pd.read_csv("all extra history.csv")

# Rename columns as you did before
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
]

# Filter out rows where title contains '#shorts' or 'music' (case-insensitive)
filtered_df = df[~df['title'].str.contains(r'#shorts|music', case=False, na=False)]

# Save to new CSV
filtered_df.to_csv("eh no shorts musics.csv", index=False)

print("Filtered CSV saved as 'eh no shorts musics.csv'")
