import sys, os
import pandas as pd
import json

# --- Fix Python import path dynamically ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import Recommender  # Import Recommender class

INPUT_FILE = "data/Gen_AI Dataset.csv"   # Could be .xlsx or .csv
OUTPUT_CSV = "data/submission.csv"

def main():
    print(f"🔍 Step 1: Loading Gen_AI Dataset from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

    # ✅ Auto-detect if it's Excel or CSV
    try:
        if INPUT_FILE.endswith(".xlsx"):
            df = pd.read_excel(INPUT_FILE, engine='openpyxl')
        else:
            df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except Exception as e:
        print("⚠️ Excel reading failed — trying as CSV instead...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    print(f"✅ Loaded {len(df)} rows from dataset")

    # Find best column name for the query
    possible_cols = ['query', 'requirement', 'problem_statement', 'text']
    query_col = next((c for c in possible_cols if c in df.columns), None)
    if query_col is None:
        raise ValueError("❌ No valid query column found in dataset!")

    print(f"🔎 Using column '{query_col}' for text input")

    print("⚙️ Step 2: Initializing Recommender...")
    recommender = Recommender(
        embeddings_path="data/catalog_embeddings.npy",
        meta_path="data/catalog_meta.json"
    )

    print("🚀 Step 3: Generating recommendations...")
    all_results = []
    for i, row in df.iterrows():
        query = str(row[query_col])
        recs = recommender.recommend_by_text(query, top_k=5)
        for r in recs:
            all_results.append({
                "input_query": query,
                "recommended_assessment": r["assessment_name"],
                "url": r["url"],
                "similarity_score": r["score"]
            })

        if (i + 1) % 10 == 0:
            print(f"   ✅ Processed {i + 1}/{len(df)} queries...")

    print("💾 Step 4: Saving results to:", OUTPUT_CSV)
    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Submission file ready → {OUTPUT_CSV}")
    print("📊 Preview:")
    print(out_df.head())

if __name__ == "__main__":
    main()
