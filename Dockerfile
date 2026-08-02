# Base Python image
FROM python:3.11-slim

# Install tesseract-ocr, the system-level OCR engine that pytesseract calls into.
# This is why we need Docker instead of Render's default Python buildpack —
# the default buildpack only installs Python packages, not system binaries like this.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the NLTK data needed to train the ML sentiment model.
# Doing this at build time (not runtime) means the server never depends
# on internet access to train the model when it starts up.
RUN python -c "import nltk; nltk.download('movie_reviews'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

# Render provides the port to bind to via the $PORT environment variable
CMD gunicorn app:app --bind 0.0.0.0:$PORT
