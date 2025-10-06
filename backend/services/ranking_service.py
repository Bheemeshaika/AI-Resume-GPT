# backend/services/ranking_service.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RankingService:
    def __init__(self, embedding_service):
        self.embedder = embedding_service

    def rank(self, job_description: str, resumes: list, top_k: int = 5):
        # create embeddings
        job_emb = self.embedder.embed(job_description)  # shape (1, dim)
        resume_embs = self.embedder.embed(resumes)      # shape (N, dim)
        # cosine similarity
        sims = cosine_similarity(job_emb.reshape(1, -1), resume_embs).flatten()
        # get sorted indices
        idx_sorted = sims.argsort()[::-1]
        results = []
        for i in idx_sorted[:top_k]:
            results.append({"index": int(i), "score": float(sims[i]), "resume_excerpt": resumes[i][:400]})
        return results
