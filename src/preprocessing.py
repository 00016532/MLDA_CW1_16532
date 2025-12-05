from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC_COLS = ["char_len","word_len","avg_word_len","sentence_count","unique_ratio"]

@dataclass(frozen=True)
class Splits:
    train_X: object
    val_X: object
    test_X: object
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray

def make_splits(df, label_col="Label", test_size=0.2, val_size=0.1, random_state=42) -> Splits:
    # First split off test
    X = df
    y = df[label_col].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then split train/val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state, stratify=y_trainval
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)

def build_preprocessor(
    tfidf_ngram=(1,2),
    tfidf_min_df=2,
    tfidf_max_df=0.9,
    tfidf_max_features=30000
) -> ColumnTransformer:
    text = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            ngram_range=tfidf_ngram,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df,
            max_features=tfidf_max_features,
        ))
    ])
    numeric = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("text", text, "Abstract"),
            ("num", numeric, NUMERIC_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre
