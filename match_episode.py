import pandas as pd
import numpy as np
import re
from pd_settings import configure_pandas_display
configure_pandas_display()

# Step 1: Load the CSV file
df = pd.read_csv("eh no shorts musics.csv")  # Replace with your actual file path
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video id", "duration", "url"
]
df["published"] = pd.to_datetime(df["published"])

# Step 2: Parse the text file and build a URL → series name mapping
series_map = {}
with open("sorted series.txt", "r", encoding="utf-8") as file:  # Replace with actual file path
    current_series = None
    for line in file:
        line = line.strip()
        if not line:
            continue
        # Detect series header, e.g., [126] Tuberculosis
        if re.match(r"\[\d+\]", line):
            match = re.match(r"\[(\d+)\]\s*(.+)", line)
            if match:
                current_series = match.group(2).strip()
        else:
            # Extract URL from video line
            if "\\" in line:
                parts = line.split("\\")
                if len(parts) > 1:
                    url = parts[1].strip()
                    series_map[url] = current_series

# Step 3: Map series names to the DataFrame using the 'URL' column
df["series name"] = df["url"].map(series_map)

df['is series'] = df['series name'] != 'Singles'

# Create 'episode_type' column
df['episode type'] = np.where(
    df['is series'],
    np.where(
        df['title'].str.contains('lies', case=False, na=False), 'Lies',
        np.where(
            df['title'].str.contains('complete', case=False, na=False), 'Complete',
            'Episode'
        )
    ),
    'Singles'
)

# Step 3: Filter only episodes
episodes_df = df[df['episode type'].isin(['Episode', 'Lies'])].copy()

# Step 4: Sort by 'series name' and 'published' date
episodes_df.sort_values(by=['series name', 'published'], inplace=True)

# # Step 5: Assign episode numbers within each series
episodes_df['episode number'] = episodes_df.groupby('series name').cumcount() + 1

# # Step 6: Merge episode numbers back into the original df using video id
episode_numbers = episodes_df[['video id', 'episode number']]
df = df.merge(episode_numbers, on='video id', how='left')

# print(df.head(100))
print(df[['views', 'likes', 'comments', 'duration', 'episode number']].dtypes)

print(df)

df.to_csv("eh_sorted.csv", index=False, encoding="utf-8-sig")