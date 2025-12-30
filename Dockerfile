FROM python:3.11-slim

WORKDIR /app

# System deps (add more if your requirements need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

EXPOSE 8080

# Healthcheck (Streamlit's internal endpoint)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8080/_stcore/health || exit 1

# Start Streamlit
CMD ["python","-m","streamlit","run","main.py", \
     "--server.port=8080","--server.address=0.0.0.0", \
     "--server.enableCORS=false","--server.enableXsrfProtection=false", \
     "--server.headless=true","--browser.gatherUsageStats=false"]
