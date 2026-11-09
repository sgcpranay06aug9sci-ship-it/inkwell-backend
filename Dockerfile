<<<<<<< HEAD
FROM python:3.11-slim

=======
# Base Python image
FROM python:3.11-slim

# Install tesseract-ocr, the system-level OCR engine that pytesseract calls into.
# This is why we need Docker instead of Render's default Python buildpack —
# the default buildpack only installs Python packages, not system binaries like this.
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

<<<<<<< HEAD
=======
# Pre-download the NLTK data needed to train the ML sentiment model.
# Doing this at build time (not runtime) means the server never depends
# on internet access to train the model when it starts up.
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
RUN python -c "import nltk; nltk.download('movie_reviews'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

<<<<<<< HEAD
# --workers 1 is REQUIRED: background jobs live in this process's memory, and
# every later request that polls for job status must reach that same memory.
# --threads lets the one worker handle uploads, polls, and background analysis
# at the same time. Requests now return quickly because the heavy work happens
# in a background thread, so the timeout only needs to cover large uploads.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
=======
# Render provides the port to bind to via the $PORT environment variable
CMD gunicorn app:app --bind 0.0.0.0:$PORT
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
