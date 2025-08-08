import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# Make sure the file is in the same directory or provide the full path
df = pd.read_csv("eh no shorts musics.csv")

df.columns = [
    "title", "published", "views", "likes", "comments",
    "video_id", "duration", "url"
]
df["published"] = pd.to_datetime(df["published"])
df["days_since_start"] = (df["published"] - df["published"].min()).dt.days
df['DayOfWeek'] = df['published'].dt.day_name()      # e.g., 'Monday'
df['WeekOfYear'] = df['published'].dt.isocalendar().week  # ISO week number

# print(df)

# To display the first few rows of the dataframe
# print(df[['title', 'published', 'DayOfWeek', 'WeekOfYear']])

# Define the correct order for days of the week
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Convert DayOfWeek to categorical with that order
df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)

# Filter out Saturdays and sort by DayOfWeek
# print(
#     df[df['DayOfWeek'] != 'Saturday']
#     [['title', 'published', 'DayOfWeek', 'WeekOfYear']]
#     .sort_values(by=['DayOfWeek', 'published'])
# )

pattern = r'(part \d)| \d | \d: |#\d|lies|complete'

# Filter rows where title does NOT match the pattern (case-insensitive)
singles = df[~df['title'].str.contains(pattern, case=False, regex=True, na=False)]
# print(singles[['title', 'published']])
# print(len(singles))

# Filter titles containing "lies" (case-insensitive)
lies_rows = df[df['title'].str.contains('lies', case=False, na=False)].copy()

# Remove everything from 'lies' onward and clean trailing non-word stuff
lies_rows['title'] = lies_rows['title'].str.replace(r'(?i)lies.*$', '', regex=True)

# Remove any leftover non-word characters (non-alphanumeric) at the end of the string
lies_rows['title'] = lies_rows['title'].str.replace(r'[\W_]+$', '', regex=True).str.strip()

# Print the cleaned titles
print(lies_rows[['title', 'published']])

titles_from_lies = lies_rows['title'].tolist()
print(titles_from_lies)
# print(df)

# print(df['title'].tolist()) 


def find_match(title):
    for t in titles_from_lies:
        if t in title:
            return t
    return None

df['series'] = df['title'].apply(find_match)

print(df[df['series'].isna()][['title', 'published', 'series']])

titles_from_lies_not_used = df['series'].unique().tolist()

print(titles_from_lies_not_used)
print(titles_from_lies)

diff = list(set(titles_from_lies_not_used) ^ set(titles_from_lies))
print(diff)
