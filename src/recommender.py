# src/recommender.py

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util


class Recommender:
    def __init__(self,
                 embeddings_path="data/catalog_embeddings.npy",
                 meta_path="data/catalog_meta.json",
                 model_name="all-MiniLM-L6-v2"):
        if not os.path.exists(embeddings_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("⚠️ Please run scripts/build_embeddings.py first.")
        
        # Load precomputed embeddings and metadata
        self.embeddings = np.load(embeddings_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        
        # Load Sentence Transformer model
        self.model = SentenceTransformer(model_name)

    def recommend_by_text(self, query_text, top_k=10):
        """Return top-k most similar assessments for a given query"""
        q_emb = self.model.encode(query_text, convert_to_numpy=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            item = self.meta[i]
            results.append({
                "assessment_name": item.get("assessment_name"),
                "url": item.get("url"),
                "score": round(float(scores[i]), 3)
            })
        return results
