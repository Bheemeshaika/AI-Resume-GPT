import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import time

# ==============================================
# 🌌 Page Config
# ==============================================
st.set_page_config(
    page_title="ResumeGPT — AI Resume Screener",
    page_icon="🌌",
    layout="centered"
)

# ==============================================
# 🎨 CSS — Background + Hover + Glass UI + Fade-in
# ==============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* ===== Page Base ===== */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background-color: #1E1E2F;
    color: white !important;
    overflow-x: hidden;
}

/* ===== Header ===== */
h1 {
    text-align: center;
    letter-spacing: -0.5px;
    font-size: 2.8em;
    font-weight: 700;
    padding-top: 1rem;
    padding-bottom: 0.5rem;
    background: -webkit-linear-gradient(45deg, #9A7CFF, #6DE4E8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(160, 120, 255, 0.5);
    transition: transform 0.3s ease, text-shadow 0.3s ease;
    display: inline-block;
    cursor: pointer;
}

h1:hover {
    transform: scale(1.05);
    text-shadow: 0 0 30px rgba(160, 120, 255, 0.7);
}

h4 {
    color: #E3E3E3 !important;
    font-weight: 600;
}

p {
    color: #D1D5DB !important;
    font-size: 1.05em;
}

/* ===== Glassmorphic Container ===== */
.main-container {
    background: rgba(30, 25, 50, 0.6);
    backdrop-filter: blur(16px) saturate(160%);
    -webkit-backdrop-filter: blur(16px) saturate(160%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 2rem;
    margin-top: 2rem;
    transition: all 0.4s ease;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
}

.main-container:hover {
    border-color: rgba(150, 80, 255, 0.8);
    box-shadow: 0 0 35px rgba(150, 80, 255, 0.5);
}

/* ===== Candidate Box ===== */
.candidate-box {
    background: rgba(50, 40, 70, 0.7);
    padding: 15px 20px;
    border-radius: 15px;
    margin: 10px 0;
    transition: all 0.5s ease, opacity 0.5s ease;
    cursor: pointer;
    opacity: 0;
}

.candidate-box.visible {
    opacity: 1;
}

.candidate-box:hover {
    background: rgba(100, 80, 200, 0.8);
    box-shadow: 0 0 20px rgba(160, 120, 255, 0.7);
    transform: translateY(-2px) scale(1.02);
}

/* ===== Stars ===== */
.stars {
    color: #FFD700;
    font-weight: 700;
}

/* ===== Text Area ===== */
[data-testid="stTextArea"] textarea {
    background: rgba(0, 0, 15, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    color: #EAEBEE !important;
    padding: 15px;
    font-size: 1rem !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.3s ease-in-out;
}

[data-testid="stTextArea"] textarea:hover {
    border-color: rgba(130, 70, 255, 0.7);
    box-shadow: 0 0 20px rgba(130, 70, 255, 0.3);
}

/* ===== File Uploader ===== */
[data-testid="stFileUploader"] section {
    background: transparent;
    border-radius: 12px;
    border: 2px dashed rgba(255, 255, 255, 0.25);
    padding: 25px;
    transition: all 0.4s ease-in-out;
}

[data-testid="stFileUploader"] section:hover {
    background: rgba(130, 70, 255, 0.1);
    border-color: rgba(130, 70, 255, 0.8);
    box-shadow: 0 0 20px rgba(130, 70, 255, 0.3);
}

/* ===== Button ===== */
.stButton > button {
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 30px;
    font-weight: 600;
    font-size: 16px;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(95, 75, 255, 0.4);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(130, 70, 255, 0.7);
    filter: brightness(1.2);
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #A0AEC0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# 🌠 Header Section
# ==============================================
st.markdown("<h1>ResumeGPT — AI Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; font-size:18px; max-width: 650px; margin: auto; padding-bottom: 1.5rem;'>
Empower your hiring with AI. Upload a job description and resumes — ResumeGPT will intelligently rank candidates by best fit.
</p>
""", unsafe_allow_html=True)

# ==============================================
# 📋 Main Interactive Container
# ==============================================


st.markdown("<h4>Documented role requirements and responsibilities</h4>", unsafe_allow_html=True)
job_description = st.text_area(
    "",
    placeholder="Paste the Job Description",
    height=150,
    label_visibility="collapsed"
)

st.markdown("<h4 style='margin-top:30px;'>Submit candidate resumes for review</h4>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Submit resumes (.txt or .pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ==============================================
# ⚡ Load SentenceTransformer model (free)
# ==============================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==============================================
# 📄 PDF Parsing
# ==============================================
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

# ==============================================
# 🧠 AI Resume Scoring Function (Free)
# ==============================================
def get_resume_score_ai(resume_text, job_desc):
    job_emb = model.encode(job_desc, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(job_emb, resume_emb).item()  # 0-1
    score = min(max(round(similarity * 5), 1), 5)  # 1-5 scale
    return score


# ==============================================
# 🏆 Process Resumes on Button Click with Loading
# ==============================================
if uploaded_files and job_description.strip():
    if st.button("Candidate Ranking"):
        with st.spinner("Analyzing resumes…"):
            rankings = []
            for file in uploaded_files:
                name = file.name.replace(".pdf", "").replace(".txt", "")
                if file.type == "text/plain":
                    resume_text = file.getvalue().decode("utf-8")
                elif file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(file)
                else:
                    resume_text = ""
                score = get_resume_score_ai(resume_text, job_description)
                rankings.append((name, score, resume_text))
            
            # Sort by score descending
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown("<h4 style='margin-top:20px;'>Candidate Rankings</h4>", unsafe_allow_html=True)
            
            for name, score, resume_text in rankings:
                stars = "★" * score + "☆" * (5 - score)
                box_html = (
                    f"<div class='candidate-box visible'>"
                    f"<strong>{name}</strong> : <span class='stars'>{stars} ({score}/5)</span><br>"
                    f"</div>"
                )
                st.markdown(box_html, unsafe_allow_html=True)
                time.sleep(0.1)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# ⚙️ Footer
# ==============================================
st.markdown("<div class='footer'>© 2025 ResumeGPT • Designed and Developed by <strong>Bheemeswararao Aika</strong></div>", unsafe_allow_html=True)
