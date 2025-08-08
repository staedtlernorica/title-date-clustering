import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def generate_ngram_matrix(text_list, n=2):
    """
    Generate an n-gram matrix from a list of strings.
    
    Args:
        text_list (list of str): Input list of strings.
        n (int): The size of n-grams to extract.
        
    Returns:
        DataFrame: A pandas DataFrame representing the n-gram frequency matrix.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word')
    ngram_matrix = vectorizer.fit_transform(text_list)
    
    df = pd.DataFrame(
        ngram_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    return df

# Example usage
texts = [
    'World War I: The Seminal Tragedy - Lies - Extra History', 'World War I: The Seminal Tragedy - The Final Act - Extra History - Part 4', 'World War I: The Seminal Tragedy - The July Crisis - Extra History - Part 3', 'World War I: The Seminal Tragedy - One Fateful Day in June - Extra History - Part 2', 'World War I: The Seminal Tragedy - The Concert of Europe - Extra History - Part 1', 'Please Support Extra History on Patreon!', 'Rome: The Punic Wars - The Conclusion of the Second Punic War - Extra History - Part 4', 'Rome: The Punic Wars - The Second Punic War Rages On - Extra History - Part 3', 'Rome: The Punic Wars - The Second Punic War Begins - Extra History - Part 2', 'Rome: The Punic Wars - The First Punic War - Extra History - Part 1']


ngram_df = generate_ngram_matrix(texts, n=2)
print(ngram_df)
