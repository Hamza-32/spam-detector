from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict

import numpy as np

from .config import SEED, TEST_SIZE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_nltk_resources() -> None:
    try:
        import nltk
    except Exception:
        return

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _get_nltk_tools():
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        return set(stopwords.words("english")), WordNetLemmatizer()
    except Exception:
        class _IdentityLemmatizer:
            def lemmatize(self, token: str) -> str:
                return token

        return set(), _IdentityLemmatizer()


def clean_text(text: str, stop_words=None, lemmatizer=None) -> str:
    if stop_words is None or lemmatizer is None:
        ensure_nltk_resources()
        stop_words, lemmatizer = _get_nltk_tools()

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]
    return " ".join(tokens)


def load_dataset(data_path: Path):
    import pandas as pd

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def add_clean_text(df):
    ensure_nltk_resources()
    stop_words, lemmatizer = _get_nltk_tools()

    out = df.copy()
    out["clean_msg"] = out["message"].apply(
        lambda msg: clean_text(msg, stop_words=stop_words, lemmatizer=lemmatizer)
    )
    return out


def split_data(df, seed: int = SEED, test_size: float = TEST_SIZE):
    from sklearn.model_selection import train_test_split

    x = df["clean_msg"]
    y = df["label_num"]

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def fit_tfidf(x_train):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    x_train_vec = tfidf.fit_transform(x_train)
    return tfidf, x_train_vec


def transform_tfidf(tfidf, x_data):
    return tfidf.transform(x_data)


def prepare_full_dataset(data_path: Path) -> Dict[str, object]:
    set_seed(SEED)
    df = load_dataset(data_path)
    df = add_clean_text(df)

    x_train, x_test, y_train, y_test = split_data(df)

    tfidf, x_train_tfidf = fit_tfidf(x_train)
    x_test_tfidf = transform_tfidf(tfidf, x_test)

    return {
        "df": df,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "tfidf": tfidf,
        "x_train_tfidf": x_train_tfidf,
        "x_test_tfidf": x_test_tfidf,
    }
