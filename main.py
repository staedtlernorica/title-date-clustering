import pandas as pd
from collections import Counter

# Pandas display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# Read your dataset
df = pd.read_csv("eh no char.csv")

# Define regex pattern to match the target phrases (case-insensitive)
pattern = r'\b(extra history|us train history|roman history|chinese history|world history|us history|british history|naval history|religious history|european history|irish history|italian history|scandinavian history|japanese history|american history|native american history|english history|egyptian history|hawaiian history|scottish history|south american history|wwi history|medical history|russian history|australian history|middle east history)\b'

# Remove the phrases from the 'title' column using regex
df['title'] = df['title'].str.replace(pattern, '', case=False, regex=True).str.replace(r'\s{2,}', ' ', regex=True).str.strip()
df.to_csv('eh.csv')

# Function to extract the most common n-gram from a list of titles
def extract_common_ngram(titles):
    cleaned_titles = [t.strip().lower() for t in titles if isinstance(t, str)]
    tokenized = [t.split(' ') for t in cleaned_titles]

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

# Function to slide over the DataFrame and run the n-gram function on each window
def sliding_window_ngram(df, window_size=20, step=5):
    n = len(df)
    for start in range(0, n, step):
        end = min(start + window_size, n)
        window_df = df.iloc[start:end]
        print(f"\n🔍 Videos from index {start} to {end-1} (total {len(window_df)}):")

        titles = window_df['title'].tolist()
        common_ngram = extract_common_ngram(titles)

        print(f"🏷️  Most common n-gram: {common_ngram}")
        print(window_df[['title', 'published']])

        if end == n:
            break

# Example usage:
# Reverse the order if needed (e.g., newest to oldest)
sliding_window_ngram(df[['title', 'published']].iloc[::-1].reset_index(drop=True), window_size=20, step=5)
