FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY example.py .

ENV PYTHONUNBUFFERED=1
ENV RATE_PER_MIN=60

EXPOSE 8000

# Health check against the liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.server:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
