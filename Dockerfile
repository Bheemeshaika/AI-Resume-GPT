# Simple Dockerfile for FastAPI backend
FROM python:3.10-slim

WORKDIR /app
COPY ./backend /app/backend
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
