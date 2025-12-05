from __future__ import annotations
import re
import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-z]+", re.I)

def add_numeric_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["Abstract"].astype(str)

    char_len = text.str.len()
    word_len = text.str.split().apply(len).replace(0, np.nan)
    tokens = text.str.lower().apply(lambda s: TOKEN_RE.findall(s))

    df["char_len"] = char_len
    df["word_len"] = word_len.fillna(0).astype(int)
    df["avg_word_len"] = (char_len / word_len).fillna(0)

    df["sentence_count"] = text.str.count(r"[.!?]") + 1
    df["unique_words"] = tokens.apply(lambda xs: len(set(xs)))
    df["unique_ratio"] = (df["unique_words"] / word_len).fillna(0)

    return df

def clip_outliers(df: pd.DataFrame, col: str = "word_len", upper_q: float = 0.995) -> pd.DataFrame:
    # keeps rows, clips extreme values to reduce outlier influence for scaling/correlation
    df = df.copy()
    if col not in df.columns:
        return df
    upper = df[col].quantile(upper_q)
    df[col] = df[col].clip(upper=upper)
    return df
