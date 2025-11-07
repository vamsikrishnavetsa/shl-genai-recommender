import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

# ============================================================
# 🔧 FIX: Patch for 'cached_download' ImportError
# ============================================================
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download
        def cached_download(*args, **kwargs):
            return hf_hub_download(*args, **kwargs)
        huggingface_hub.cached_download = cached_download
except Exception as e:
    print("⚠️ HuggingFace Hub patch failed:", e)

from sentence_transformers import SentenceTransformer, util

# ============================================================
# ⚙️ Configuration
# ============================================================
EMBED_PATH = "data/catalog_embeddings.npy"
META_PATH = "data/catalog_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================
# 🧠 Load Model & Data
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the SentenceTransformer model"""
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def load_data():
    """Load embeddings and metadata"""
    if not os.path.exists(EMBED_PATH) or not os.path.exists(META_PATH):
        st.error("❌ Required data files not found. Please check your 'data/' folder.")
        st.stop()

    try:
        embeddings = np.load(EMBED_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading embeddings or metadata: {e}")
        st.stop()
    return embeddings, meta


# Initialize
model = load_model()
embeddings, meta = load_data()

# ============================================================
# 🖥️ Streamlit Page Configuration
# ============================================================
st.set_page_config(
    page_title="SHL GenAI Recommender",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 🎨 Header
# ============================================================
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>🤖 <b>SHL GenAI Assessment Recommender</b></h1>
        <p style='font-size:18px;'>
            Type any <b>skill</b>, <b>role</b>, or <b>competency</b> and instantly discover 
            the most relevant <b>SHL assessments</b> powered by 
            <span style='color:#7A42F4;'>GenAI</span> 🚀
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 🔍 Query Input Section
# ============================================================
st.markdown("---")

query = st.text_input("🔍 Enter your query (e.g., 'communication skills', 'leadership test'):")

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Top N Results", 1, 10, 5)
with col2:
    st.write(" ")

# ============================================================
# ⚡ Recommendation Logic
# ============================================================
if st.button("✨ Generate Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first ❗")
    else:
        with st.spinner("🔎 Analyzing and matching best SHL assessments..."):
            try:
                q_emb = model.encode(query, convert_to_numpy=True)
                scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
                top_idx = np.argsort(scores)[::-1][:top_k]

                results = []
                for i in top_idx:
                    item = meta[i]
                    results.append({
                        "Assessment Name": item.get("assessment_name", "N/A"),
                        "Category": item.get("category", "N/A"),
                        "URL": item.get("url", "N/A"),
                        "Similarity Score": round(float(scores[i]), 3),
                    })

                df = pd.DataFrame(results)

                st.success(f"✅ Top {top_k} recommendations for: **{query}**")
                st.markdown("---")

                # Display results
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # Highlight best match
                top_item = df.iloc[0]
                st.markdown(
                    f"""
                    <div style='padding:15px; background-color:#f8f9fa; border-radius:10px; border-left:5px solid #7A42F4;'>
                        <h4>🏆 Best Match: {top_item['Assessment Name']}</h4>
                        <p><b>Category:</b> {top_item['Category']}</p>
                        <p><b>Similarity Score:</b> {top_item['Similarity Score']}</p>
                        <a href="{top_item['URL']}" target="_blank">🔗 View Assessment</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"⚠️ Error generating recommendations: {e}")

# ============================================================
# 🦶 Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; color:gray;'>
        Built with ❤️ by <b>Vamsi Krishna Vetsa</b> | Powered by <b>SHL GenAI</b> ⚡
    </p>
    """,
    unsafe_allow_html=True,
)
