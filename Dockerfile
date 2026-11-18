FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY app.py agent.py gemini.py risk.py ./
COPY data/ ./data/

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read().strip() == b'ok'"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
