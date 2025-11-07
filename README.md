---
title: SHL GenAI Recommender
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: app_combined.py
pinned: false
---
🧠 SHL GenAI Assessment Recommender System

An AI-powered semantic recommendation engine that suggests the most relevant SHL assessments for any given skill, role, or competency query — built with FastAPI, Streamlit, and Sentence Transformers.

🚀 Overview

The SHL GenAI Recommender intelligently maps a user’s input (e.g., “communication skills” or “leadership test”) to relevant SHL assessments.
It uses Transformer-based sentence embeddings (all-MiniLM-L6-v2) to compute semantic similarity between the user’s query and pre-encoded SHL assessment data.

🏗️ Project Architecture

User Query
   ↓
Streamlit UI (Frontend)
   ↓
FastAPI (Backend)
   ↓
Recommender Engine
   ↓
Embeddings Database (.npy + .json)
   ↓
Top N Assessment Recommendations

⚙️ Features

✅ Semantic search using transformer embeddings
✅ FastAPI backend for lightning-fast responses
✅ Streamlit UI for easy interaction
✅ Precomputed embeddings for quick similarity search
✅ Scalable and modular architecture

🧰 Tech Stack

| Component | Technology                                 |
| --------- | ------------------------------------------ |
| Language  | Python 3.11                                |
| ML Model  | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Backend   | FastAPI                                    |
| Frontend  | Streamlit                                  |
| Server    | Uvicorn                                    |
| Libraries | NumPy, Pandas, Transformers, Requests      |
| Storage   | CSV, JSON, NPY                             |

📂 Project Structure

SHL-GenAI-Recommender/
│
├── data/
│   ├── shl_catalog.csv
│   ├── catalog_embeddings.npy
│   ├── catalog_meta.json
│   └── Gen_AI Dataset.csv
│
├── scripts/
│   ├── build_embeddings.py       # Generates embeddings for catalog data
│   ├── generate_submission.py    # Generates submission results for evaluation
│
├── src/
│   ├── app.py                    # FastAPI backend
│   └── recommender.py            # Core recommender logic
│
├── app_ui.py                     # Streamlit frontend
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
└── SHL_GenAI_Recommender_Project_Vamsi.pdf  # Final report

🧪 How to Run
1️⃣ Clone the Repository
git clone https://github.com/yourusername/genai-recommender.git
cd genai-recommender

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate 

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Build Embeddings
python scripts/build_embeddings.py

5️⃣ Run FastAPI Backend
uvicorn src.app:app --reload

6️⃣ Launch Streamlit Frontend
streamlit run app_ui.py

Access app at: http://localhost:8501

🧩 Example Query
Input:
Recommend a test to evaluate communication skills
Output:
| Rank | Assessment Name               | Similarity Score |
| ---- | ----------------------------- | ---------------- |
| 1    | Communication Skills Test     | 0.498            |
| 2    | Teamwork and Collaboration    | 0.475            |
| 3    | Verbal Interaction Assessment | 0.462            |

📈 Results

Average API response time: < 200ms

Top 5 semantic matches displayed in under 1s

90%+ accuracy for relevant test suggestions

🧭 Future Improvements

Add Retrieval-Augmented Generation (RAG) for explainable recommendations

Integrate feedback-based model refinement

Deploy to AWS / Hugging Face Spaces

Add support for multilingual queries

👨‍💻 Author

Vetsa Vamsi Krishna
B.Tech IT, Batch 2026
📧 Email: vetsavamsi@gmail.com
