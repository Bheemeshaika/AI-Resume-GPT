# Resume GPT - AI Resume Screener

ResumeGPT is an AI-driven resume analysis application that helps recruiters and job seekers evaluate resumes against specific job descriptions.
Built using Streamlit, Python, and OpenAI APIs, the app automatically compares uploaded resumes, extracts key skills, and ranks candidates based on job relevance. It also provides visual indicators such as rating stars, hover effects, and interactive UI for better user experience.

Key Features:

🧠 AI-based resume scoring and ranking

📊 Real-time candidate comparison

✨ Interactive UI with hover effects and colored star ratings

📁 Multi-resume upload support

💬 Smart job description parsing and matching

⚙️ Built with Streamlit and OpenAI API

# Steps to be Followed
## Quick Local Run
## Quick local run (recommended) after cloning the repository

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
