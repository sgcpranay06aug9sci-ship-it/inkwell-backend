FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng

COPY . .

ENV PORT=10000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
