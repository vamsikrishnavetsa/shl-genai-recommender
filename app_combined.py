import threading
import time
import requests
import subprocess
import streamlit as st

# Step 1: Start FastAPI in background
def start_fastapi():
    subprocess.Popen(["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"])

# Step 2: Start the FastAPI server
threading.Thread(target=start_fastapi, daemon=True).start()
time.sleep(5)

# Step 3: Streamlit UI
st.set_page_config(page_title="SHL GenAI Recommender", page_icon="🤖")

st.title("🤖 SHL GenAI Assessment Recommender")
st.write("Type any skill, role, or competency — and get SHL assessments instantly powered by GenAI!")

query = st.text_input("🔍 Enter your query (e.g. 'communication skills')", "")
top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        try:
            res = requests.post("http://localhost:8000/recommend", json={"query": query, "top_k": top_k})
            if res.status_code == 200:
                results = res.json()
                st.success(f"Top {len(results)} Recommendations:")
                for r in results:
                    st.write(f"**{r['assessment_name']}** — Similarity: {r['score']}")
                    st.write(f"[Link]({r['url']})")
                    st.write("---")
            else:
                st.error(f"Error from API: {res.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
