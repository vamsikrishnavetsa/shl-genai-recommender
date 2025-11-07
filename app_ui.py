import streamlit as st
import requests
import pandas as pd

# ------------------------------
# 🔧 Configuration
# ------------------------------
API_URL = "http://127.0.0.1:8000/recommend"  # FastAPI backend endpoint

st.set_page_config(page_title="SHL GenAI Recommender", layout="centered")

# ------------------------------
# 🎨 Page Title
# ------------------------------
st.title("🤖 SHL GenAI Assessment Recommender")
st.write("Type any skill, role, or competency — and get SHL assessments instantly powered by GenAI!")

# ------------------------------
# 🧠 Input Section
# ------------------------------
query = st.text_input("🔍 Enter your query (e.g. 'communication skills' or 'leadership test')", "")
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("🚀 Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                payload = {"query": query, "top_k": top_k}
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    recs = data.get("recommendations", [])

                    if len(recs) == 0:
                        st.info("No recommendations found.")
                    else:
                        st.success(f"✅ Found {len(recs)} recommendations for: **{query}**")

                        # Display results as table
                        df = pd.DataFrame(recs)
                        st.dataframe(df, use_container_width=True)

                        # Optional: show cards
                        st.markdown("---")
                        for i, rec in enumerate(recs, start=1):
                            st.markdown(f"""
                                **{i}. [{rec['assessment_name']}]({rec['url']})**  
                                🔹 *Similarity:* `{rec['score']}`
                                """)
                else:
                    st.error(f"API Error: {response.status_code} — {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

# ------------------------------
# ℹ️ Footer
# ------------------------------
st.markdown("---")
st.caption("Built by Vamsi ⚡ — powered by SHL GenAI Recommender")
