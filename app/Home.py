from app._pathfix import add_project_root_to_path
add_project_root_to_path(__file__)

import streamlit as st

st.set_page_config(page_title="Inflation Abstracts Classifier", layout="wide")

st.title("Inflation Abstracts — Coursework App")
st.write("Multi‑page Streamlit app: **EDA → Preprocessing → Train & Compare → Inference**")

st.markdown("### Run commands")
st.code("python -m src.train\npython -m streamlit run app/Home.py")

st.info("Dataset expected at: data/raw/classified_abstracts.json")
