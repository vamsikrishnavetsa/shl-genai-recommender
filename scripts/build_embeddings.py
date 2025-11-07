# scripts/build_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import sys

# === Configuration ===
CSV_PATH = "data/shl_catalog.csv"          # Input catalog (from SHL dataset)
OUT_EMBED = "data/catalog_embeddings.npy"  # Output embeddings file
OUT_META = "data/catalog_meta.json"        # Output metadata file
MODEL_NAME = "all-MiniLM-L6-v2"            # SentenceTransformer model

def main():
    print("🔍 Step 1: Loading catalog from", CSV_PATH)

    # --- Check file existence ---
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: File not found at {CSV_PATH}")
        sys.exit(1)

    # --- Load dataset ---
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        sys.exit(1)

    # --- Validate columns ---
    required_cols = ['assessment_name', 'category', 'description']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Missing column '{col}' in dataset. Filling with blanks.")
            df[col] = ""

    # --- Combine relevant text fields ---
    df["text"] = (
        df["assessment_name"].fillna("") + ". " +
        df["category"].fillna("") + ". " +
        df["description"].fillna("")
    )

    texts = df["text"].astype(str).tolist()
    print(f"✅ Loaded {len(texts)} entries for embedding.")

    # --- Load model ---
    print(f"📦 Step 2: Loading SentenceTransformer model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)

    # --- Generate embeddings ---
    print("⚙️ Step 3: Generating text embeddings (this may take a few minutes)...")
    try:
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        sys.exit(1)

    # --- Save outputs ---
    os.makedirs("data", exist_ok=True)
    np.save(OUT_EMBED, embeddings)

    # Save metadata (exclude embeddings)
    meta_cols = [c for c in df.columns if c != "text"]
    meta = df[meta_cols].to_dict(orient="records")
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n✅ Step 4: Completed successfully!")
    print(f"   → Embeddings saved to: {OUT_EMBED}")
    print(f"   → Metadata saved to:   {OUT_META}\n")

if __name__ == "__main__":
    main()
