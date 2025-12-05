from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from src.paths import DATA_RAW, DATA_PROCESSED

def load_raw_json(path: Path = DATA_RAW) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    # Required columns expected: DOI, Abstract, Label
    df["Label"] = df["Label"].astype(int)
    df["DOI"] = df.get("DOI", pd.Series([None]*len(df))).fillna("unknown").astype(str)
    df["Abstract"] = df["Abstract"].astype(str)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Abstract"] = (
        df["Abstract"]
        .str.replace(r"^abstract[:\s]+", "", regex=True, flags=re.I)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["Abstract"].str.len() > 0].copy()
    # Remove duplicates to reduce leakage risk (do this BEFORE splitting)
    df = df.drop_duplicates(subset=["Abstract"]).reset_index(drop=True)
    return df

def save_processed(df: pd.DataFrame, path: Path = DATA_PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_processed_or_build() -> pd.DataFrame:
    if DATA_PROCESSED.exists():
        df = pd.read_csv(DATA_PROCESSED)
        df["Label"] = df["Label"].astype(int)
        df["DOI"] = df.get("DOI", pd.Series([None]*len(df))).fillna("unknown").astype(str)
        df["Abstract"] = df["Abstract"].astype(str)
        df = df.drop_duplicates(subset=["Abstract"]).reset_index(drop=True)
        return df

    df = basic_clean(load_raw_json())
    save_processed(df)
    return df
