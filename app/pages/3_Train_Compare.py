from app._pathfix import add_project_root_to_path
add_project_root_to_path(__file__)

import streamlit as st
import json
from pathlib import Path

from src.data import load_processed_or_build
from src.features import add_numeric_text_features, clip_outliers
from src.preprocessing import make_splits
from src.modeling import build_model_pipelines
from src.evaluation import evaluate

from sklearn.model_selection import GridSearchCV

st.title("Train & Compare")

df = load_processed_or_build()
df = add_numeric_text_features(df)
df = clip_outliers(df, col="word_len", upper_q=0.995)

splits = make_splits(df, label_col="Label", test_size=0.2, val_size=0.1, random_state=42)

mode = st.selectbox("Mode", ["Quick (fit default)", "GridSearch (macro-F1)"], index=0)

if st.button("Run"):
    results = {}
    for name, (pipe, grid) in build_model_pipelines().items():
        st.write(f"Training: **{name}**")
        if mode.startswith("Quick"):
            pipe.fit(splits.train_X, splits.train_y)
            model = pipe
            results[name] = {
                "val": evaluate(model, splits.val_X, splits.val_y),
                "test": evaluate(model, splits.test_X, splits.test_y),
                "best_params": "default"
            }
        else:
            search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=5, n_jobs=-1)
            search.fit(splits.train_X, splits.train_y)
            model = search.best_estimator_
            results[name] = {
                "val": evaluate(model, splits.val_X, splits.val_y),
                "test": evaluate(model, splits.test_X, splits.test_y),
                "best_params": search.best_params_
            }

    st.subheader("Results")
    st.json(results)

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/streamlit_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    st.success("Saved artifacts/streamlit_results.json")
