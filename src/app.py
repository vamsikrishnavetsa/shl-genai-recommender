from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.recommender import Recommender

# Initialize FastAPI app
app = FastAPI(
    title="SHL GenAI Recommender API",
    description="API that recommends SHL assessments based on natural language queries.",
    version="1.0"
)

# Set up templates directory (for frontend)
templates = Jinja2Templates(directory="src/templates")

# Load recommender model safely
try:
    recommender = Recommender()
except Exception as e:
    raise RuntimeError(f"Failed to initialize recommender: {e}")

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# 🏠 Web UI (Home page)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Renders the simple HTML page for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# ❤️ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SHL Gen AI API is live. Use `/health` and `/recommend`."}

# 🚀 Main recommendation endpoint
@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    try:
        results = recommender.recommend_by_text(request.query, request.top_k)
        return {"query": request.query, "recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
