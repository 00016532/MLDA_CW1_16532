from app._pathfix import add_project_root_to_path
add_project_root_to_path(__file__)

import streamlit as st
import joblib
import pandas as pd
from src.paths import MODEL_PATH
from src.features import add_numeric_text_features

st.title("Inference")

if not MODEL_PATH.exists():
    st.warning("Model not found. Train first: `python -m src.train`")
    st.stop()

model = joblib.load(MODEL_PATH)

text = st.text_area("Paste abstract:", height=220)
threshold = st.slider("Threshold (class 1)", 0.10, 0.90, 0.50, 0.05)

if st.button("Predict"):
    if not text.strip():
        st.error("Empty text.")
        st.stop()

    df = pd.DataFrame([{"DOI":"unknown","Abstract":text,"Label":0}])
    df = add_numeric_text_features(df)
    X = df  # model expects dataframe with Abstract + numeric cols

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        p1 = float(proba[1])
        pred = 1 if p1 >= threshold else 0
        st.write({"p0": float(proba[0]), "p1": p1})
        st.subheader(f"Prediction: {pred}")
    else:
        pred = int(model.predict(X)[0])
        st.subheader(f"Prediction: {pred}")
