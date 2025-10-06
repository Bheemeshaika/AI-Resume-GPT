# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer


# ==============================================
# 🌐 FastAPI Setup
# ==============================================
app = FastAPI(title="ResumeGPT Backend")

# Allow CORS for frontend
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# 📦 Models
# ==============================================
class RankRequest(BaseModel):
    job_description: str
    resumes: List[str]

class AskRequest(BaseModel):
    question: str
    job_description: str
    resumes: List[str]

# ==============================================
# 🛠 Sentence Transformer Model (Offline Embeddings)
# ==============================================
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good for semantic similarity

def real_embedding(text):
    """Return semantic embedding vector for a given text"""
    return model.encode(text)

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ==============================================
# 🛠 Ranking Function
# ==============================================
def rank_candidates(job_description, resumes):
    jd_vec = real_embedding(job_description)
    results = []
    for i, r in enumerate(resumes):
        r_vec = real_embedding(r)
        score = cosine_similarity(jd_vec, r_vec)
        # Convert similarity (0-1) to rating out of 5
        rating = round(float(score) * 5, 2)
        results.append({"index": i, "rating": rating})
    # Sort by rating descending
    results = sorted(results, key=lambda x: x["rating"], reverse=True)
    return results


# ==============================================
# 🚀 Endpoints
# ==============================================
@app.post("/rank")
def rank_endpoint(req: RankRequest, top_k: int = 5):
    try:
        ranked_results = rank_candidates(req.job_description, req.resumes)
        return {"top_k": min(top_k, len(ranked_results)), "results": ranked_results[:top_k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


