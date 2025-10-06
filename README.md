ac# ResumeGPT - AI Resume Screener (Starter)
This repository contains a starter scaffold for **ResumeGPT**, an AI Resume Screening & Job Fit Analyzer using free tools.

## What's included
- FastAPI backend (`backend/`) for embeddings + ranking
- Streamlit frontend (`frontend/`) for uploading job descriptions and resumes
- Example scripts and minimal code to run locally using free tools (Sentence-Transformers + FAISS or numpy similarity)
- `requirements.txt` listing the Python packages to install

## Quick local run (recommended)
1. Create and activate a Python venv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start backend:
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```
3. Start frontend (in another terminal):
   ```bash
   streamlit run frontend/app.py
   ```

## Notes
- This is a starter scaffold designed to use **free** OSS tools (Sentence-Transformers, FAISS local or numpy).
- For production-grade usage, follow the Implementation Plan in the PDF (`ResumeGPT_FreeTools_and_ImplementationPlan.pdf`) included in this package.
