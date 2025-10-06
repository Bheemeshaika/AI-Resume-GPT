# backend/models/resume_parser.py
# Minimal resume parser - for real projects consider using Apache Tika, pdfminer or Grobid for richer parsing.
def extract_text_from_txt_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
