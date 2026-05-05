from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def create_vectorizer(max_features=10000, ngram_range=(1, 1)):
    """
    Initializes a TF-IDF vectorizer.
    max_features: maximum number of features (vocabulary size)
    ngram_range: tuple, e.g., (1,1) for unigrams, (1,2) for unigrams+bigrams
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    return vectorizer


def fit_vectorizer(vectorizer, text_data):
    """
    Learns vocabulary and IDF from text_data.
    text_data: iterable of strings (e.g., list or Series)
    """
    vectorizer.fit(text_data)
    return vectorizer


def transform_text(vectorizer, text_data):
    """
    Converts text_data into TF-IDF vectors using a fitted vectorizer.
    Returns a sparse matrix.
    """
    return vectorizer.transform(text_data)


def fit_transform_text(vectorizer, text_data):
    """
    Fits the vectorizer to text_data and transforms it in one step.
    Returns a sparse matrix.
    """
    return vectorizer.fit_transform(text_data)


def save_vectorizer(vectorizer, path):
    """
    Saves the trained vectorizer using joblib.
    """
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to {path}")


def load_vectorizer(path):
    """
    Loads a saved vectorizer.
    """
    vectorizer = joblib.load(path)
    print(f"Vectorizer loaded from {path}")
    return vectorizer
