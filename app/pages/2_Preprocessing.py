from app._pathfix import add_project_root_to_path
add_project_root_to_path(__file__)

import streamlit as st
from src.data import load_processed_or_build
from src.features import add_numeric_text_features, clip_outliers
from src.preprocessing import build_preprocessor, NUMERIC_COLS

st.title("Preprocessing")

df = load_processed_or_build()
df = add_numeric_text_features(df)
df = clip_outliers(df, col="word_len", upper_q=0.995)

st.markdown("### TF‑IDF settings")
ngram = st.selectbox("ngram_range", [(1,1),(1,2)], index=1)
min_df = st.slider("min_df", 1, 10, 2, 1)
max_df = st.slider("max_df", 0.70, 1.00, 0.90, 0.01)
max_features = st.selectbox("max_features", [None, 10000, 30000, 60000], index=2)

pre = build_preprocessor(tfidf_ngram=ngram, tfidf_min_df=min_df, tfidf_max_df=max_df, tfidf_max_features=max_features)

X = df[["Abstract"] + NUMERIC_COLS]
pre.fit(X)

Xt = pre.transform(X)
st.write("Transformed shape:", Xt.shape)
st.write("Numeric columns:", NUMERIC_COLS)
