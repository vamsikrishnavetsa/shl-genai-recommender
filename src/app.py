from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.recommender import Recommender

# Initialize FastAPI app
app = FastAPI(
    title="SHL GenAI Recommender API",
    description="API that recommends SHL assessments based on natural language queries.",
    version="1.0"
)

# Load recommender model
try:
    recommender = Recommender()
except Exception as e:
    raise RuntimeError(f"Failed to initialize recommender: {e}")

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Root health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}

# Main recommendation endpoint
@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    try:
        results = recommender.recommend_by_text(request.query, request.top_k)
        return {"query": request.query, "recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
