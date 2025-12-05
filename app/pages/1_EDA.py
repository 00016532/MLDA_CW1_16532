from app._pathfix import add_project_root_to_path
add_project_root_to_path(__file__)

import streamlit as st
import matplotlib.pyplot as plt
from src.data import load_processed_or_build
from src.features import add_numeric_text_features, clip_outliers
from src.preprocessing import NUMERIC_COLS

st.title("EDA")

df = load_processed_or_build()
df = add_numeric_text_features(df)
df = clip_outliers(df, col="word_len", upper_q=0.995)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Label 0", int((df["Label"]==0).sum()))
c3.metric("Label 1", int((df["Label"]==1).sum()))

st.subheader("Preview")
st.dataframe(df[["DOI","Label","Abstract"]].head(15), use_container_width=True)

st.subheader("Numeric feature summary")
st.dataframe(df[NUMERIC_COLS].describe().T, use_container_width=True)

st.subheader("Class balance")
counts = df["Label"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(counts.index.astype(str), counts.values)
ax.set_xlabel("Label")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Length distribution (words)")
fig2, ax2 = plt.subplots()
ax2.hist(df["word_len"].values, bins=40)
ax2.set_xlabel("Words")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

st.subheader("Correlation matrix (numeric features + label)")
corr_cols = NUMERIC_COLS + ["Label"]
corr = df[corr_cols].corr(numeric_only=True)

fig3, ax3 = plt.subplots(figsize=(7,5))
im = ax3.imshow(corr.values, interpolation="nearest")
ax3.set_xticks(range(len(corr.columns)))
ax3.set_xticklabels(corr.columns, rotation=45, ha="right")
ax3.set_yticks(range(len(corr.index)))
ax3.set_yticklabels(corr.index)
fig3.colorbar(im)
st.pyplot(fig3)
