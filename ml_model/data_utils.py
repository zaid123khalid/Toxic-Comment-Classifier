import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)


def load_dataset(path):
    """Loads the dataset from the specified path."""
    if not os.path.exists(path):
        print(f"Error: The file {path} does not exist.")
        return None
    try:
        data = pd.read_csv(path)
        print(f"Dataset loaded successfully from {path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def inspect_dataset(df):
    """Prints basic information about the dataset."""
    if df is None:
        print("No dataset to inspect.")
        return

    print("Dataset Shape:", df.shape)

    print("Dataset Info:")
    df.info()

    print("\nFirst 5 Rows:")
    print(df.head())

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    for label in labels:
        print(df[label].value_counts())
        print()

    df["comment_length"] = df["comment_text"].apply(lambda x: len(str(x).split()))
    print("\nComment Length Statistics:")
    print(df["comment_length"].describe())


def get_features_and_labels(df, label_column="toxic"):
    """
    Separates features and labels from the dataset.

    Args:
        df: DataFrame with text and label columns
        label_column: str or list - single label name or list of labels
                      Default: "toxic" for binary classification
    """
    if df is None:
        print("No dataset to extract features and labels from.")
        return None, None

    X = df["comment_text"]

    if isinstance(label_column, str):
        y = df[label_column]
    else:
        y = df[label_column]

    return X, y


def clean_text(text):
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = normalize_text(text)
    return text


def remove_urls(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    return text


def remove_punctuation(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_numbers(text):
    text = re.sub(r"\d+", "", text)
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])


def normalize_text(text):
    return text.lower().strip()


def preprocess_dataframe(df):
    df["comment_text"] = df["comment_text"].apply(clean_text)
    return df


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
