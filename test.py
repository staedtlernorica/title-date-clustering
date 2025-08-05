import re
from collections import defaultdict
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

INPUT_CSV = "eh no shorts musics.csv"

df = pd.read_csv(INPUT_CSV)
df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
    ]

import re
import pandas as pd

# Example DataFrame: Replace this with your own df
# df = pd.DataFrame({'date': pd.to_datetime([...])})

# Make sure the 'date' column is datetime
df['published'] = pd.to_datetime(df['published'])

window_size = 15

# Sort df by date, just in case
df = df.sort_values('published').reset_index(drop=True)

# Store the spans and indices of windows
spans = []

for start in range(len(df) - window_size + 1):
    window_dates = df.loc[start:start + window_size - 1, 'published']
    span = window_dates.max() - window_dates.min()
    spans.append((span, start))

# Find the 20-row window with shortest span
shortest_span, shortest_start = min(spans, key=lambda x: x[0])

# Find the 20-row window with longest span
longest_span, longest_start = max(spans, key=lambda x: x[0])

print(f"Shortest span: {shortest_span}, starting at row {shortest_start}")
print(f"Longest span: {longest_span}, starting at row {longest_start}")

# Extract those entries
shortest_entries = df.loc[shortest_start:shortest_start + window_size - 1]
longest_entries = df.loc[longest_start:longest_start + window_size - 1]

print("Shortest span entries:")
print(shortest_entries)

print("\nLongest span entries:")
print(longest_entries)



# df.to_csv("eh no char.csv", index=False)
