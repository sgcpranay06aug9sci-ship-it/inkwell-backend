FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('movie_reviews'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

# --workers 1 is REQUIRED: background jobs live in this process's memory, and
# every later request that polls for job status must reach that same memory.
# --threads lets the one worker handle uploads, polls, and background analysis
# at the same time. Requests now return quickly because the heavy work happens
# in a background thread, so the timeout only needs to cover large uploads.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
