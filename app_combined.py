import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Load prebuilt data
# ----------------------------
EMBED_PATH = "data/catalog_embeddings.npy"
META_PATH = "data/catalog_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    embeddings = np.load(EMBED_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return embeddings, meta

model = load_model()
embeddings, meta = load_data()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SHL GenAI Recommender", page_icon="🤖", layout="centered")

st.title("🤖 SHL GenAI Assessment Recommender")
st.markdown("Type any skill, role, or competency — and get SHL assessments instantly powered by GenAI!")

query = st.text_input("🔍 Enter your query (e.g., communication skills, leadership test):", "")
top_k = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations 🚀"):
    if not query.strip():
        st.warning("Please enter a query first!")
    else:
        st.write("Generating recommendations, please wait...")

        q_emb = model.encode(query, convert_to_numpy=True)
        scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            item = meta[i]
            results.append({
                "assessment_name": item.get("assessment_name"),
                "url": item.get("url"),
                "similarity": round(float(scores[i]), 3)
            })

        df = pd.DataFrame(results)
        st.success(f"Top {top_k} SHL Assessments for '{query}':")
        st.dataframe(df)
