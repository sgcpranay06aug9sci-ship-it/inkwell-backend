"""
app.py — Inkwell backend (v2: adds PDF and Photo/OCR upload support)
----------------------------------------------------------------------
Existing features:
  1. Sentence-level sentiment analysis (lexicon-based)
  2. Extractive summarization (word-frequency scoring)
  3. Keyword extraction (frequency-based, stopwords removed)
  4. Basic text statistics (word count, reading time)

New in this version:
  5. PDF upload -> extract text -> run through the same analysis
  6. Photo upload -> OCR (extract text from image) -> run through the same analysis
"""

import re
import io
from collections import Counter
from flask import Flask, request, jsonify

from pypdf import PdfReader
from PIL import Image
import pytesseract

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

# ---------------------------------------------------------------
# ML SENTIMENT MODEL (trained once at server startup, not per-request)
# ---------------------------------------------------------------
# This is a real trained Naive Bayes classifier from TextBlob, trained on
# a labeled corpus of 2,000 movie reviews (nltk's `movie_reviews` corpus).
# We train it once here so every request afterward is instant.
print("Training ML sentiment model (one-time, ~5 seconds)...")
ML_ANALYZER = NaiveBayesAnalyzer()
TextBlob("warming up the model", analyzer=ML_ANALYZER).sentiment  # triggers training
print("ML sentiment model ready.")


def ml_sentiment(text):
    """Real ML-based sentiment using a trained Naive Bayes classifier."""
    result = TextBlob(text, analyzer=ML_ANALYZER).sentiment
    confidence = round(max(result.p_pos, result.p_neg) * 100)
    label = "positive" if result.classification == "pos" else "negative"
    return {"label": label, "confidence": confidence}


@app.after_request
def add_cors_headers(response):
    """Allow the mobile app (running on a phone, a different origin) to call this API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------
# 1. STATIC RESOURCES (no downloads needed — everything is local)
# ---------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself just let's me more most mustn't my myself no nor not of off on
once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves
""".split())

POSITIVE_WORDS = set("""
good great excellent amazing awesome fantastic wonderful love loved lovely
happy joy delightful positive best better brilliant beautiful nice perfect
outstanding impressive superb favorite fun exciting glad pleased satisfied
success successful helpful improve improved improvement recommend recommended
enjoy enjoyed enjoying easy smooth reliable efficient valuable useful
""".split())

NEGATIVE_WORDS = set("""
bad terrible awful horrible hate hated worst poor disappointing disappointed
negative sad angry annoying frustrating frustrated broken fail failed failure
problem problems issue issues difficult hard confusing slow buggy bug crash
crashed useless waste wasted worse worthless boring painful complain complaint
""".split())


# ---------------------------------------------------------------
# 2. TEXT PROCESSING HELPERS
# ---------------------------------------------------------------

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_sentiment(sentence):
    words = tokenize(sentence)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = pos - neg
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return label, score


def overall_sentiment(sentences_data):
    if not sentences_data:
        return {"label": "neutral", "positive": 0, "negative": 0, "neutral": 0}
    counts = Counter(s["label"] for s in sentences_data)
    total = len(sentences_data)
    if counts["positive"] > counts["negative"]:
        label = "positive"
    elif counts["negative"] > counts["positive"]:
        label = "negative"
    else:
        label = "neutral"
    return {
        "label": label,
        "positive": round(counts["positive"] / total * 100),
        "negative": round(counts["negative"] / total * 100),
        "neutral": round(counts["neutral"] / total * 100),
    }


def extract_keywords(text, top_n=8):
    words = tokenize(text)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def summarize(text, sentences, num_sentences=3):
    if len(sentences) <= num_sentences:
        return sentences

    words = tokenize(text)
    filtered = [w for w in words if w not in STOPWORDS]
    freq = Counter(filtered)
    max_freq = max(freq.values()) if freq else 1
    normalized = {w: c / max_freq for w, c in freq.items()}

    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_words = tokenize(sentence)
        score = sum(normalized.get(w, 0) for w in sentence_words)
        if idx == 0:
            score *= 1.2
        scored.append((idx, sentence, score))

    top = sorted(scored, key=lambda x: x[2], reverse=True)[:num_sentences]
    top_in_order = sorted(top, key=lambda x: x[0])
    return [s[1] for s in top_in_order]


def text_stats(text, sentences):
    words = tokenize(text)
    word_count = len(words)
    char_count = len(text)
    reading_time_min = max(1, round(word_count / 200))
    return {
        "words": word_count,
        "characters": char_count,
        "sentences": len(sentences),
        "reading_time": reading_time_min,
    }


def count_syllables(word):
    """Rough syllable estimate: count vowel groups. Good enough for readability scoring."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_score(text, words, sentences):
    """
    Flesch-Kincaid Grade Level — a standard, named readability formula:
    0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    Returns an approximate US school grade level needed to understand the text.
    """
    if not words or not sentences:
        return {"grade_level": 0, "label": "N/A"}

    syllable_count = sum(count_syllables(w) for w in words)
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllable_count / len(words)) - 15.59
    grade = max(0, round(grade, 1))

    if grade <= 6:
        label = "Easy (elementary level)"
    elif grade <= 9:
        label = "Moderate (middle school level)"
    elif grade <= 13:
        label = "Difficult (high school / college level)"
    else:
        label = "Very difficult (graduate level)"

    return {"grade_level": grade, "label": label}


def word_frequencies(text, top_n=25):
    """Word frequency data for the word cloud visualization."""
    words = tokenize(text)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(filtered)
    return [{"word": w, "count": c} for w, c in counts.most_common(top_n)]


def build_analysis(text):
    """Shared analysis pipeline used by both the text endpoint and the file endpoint."""
    sentences = split_sentences(text)
    words = tokenize(text)
    sentences_data = []
    for s in sentences:
        label, score = sentence_sentiment(s)
        sentences_data.append({"text": s, "label": label, "score": score})

    return {
        "sentiment": overall_sentiment(sentences_data),
        "ml_sentiment": ml_sentiment(text),
        "sentences": sentences_data,
        "keywords": extract_keywords(text),
        "word_cloud": word_frequencies(text),
        "summary": summarize(text, sentences),
        "stats": text_stats(text, sentences),
        "readability": readability_score(text, words, sentences),
    }


# ---------------------------------------------------------------
# 3. FILE TEXT EXTRACTION (NEW)
# ---------------------------------------------------------------

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()


# ---------------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "Inkwell backend is running",
        "endpoints": {
            "/api/analyze": "POST { text } -> analysis",
            "/api/analyze-file": "POST multipart file (pdf/png/jpg) -> analysis",
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text is too long. Please limit to 20,000 characters."}), 400

    result = build_analysis(text)
    return jsonify(result)


@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]
    filename = (file.filename or "").lower()

    if not filename:
        return jsonify({"error": "Uploaded file has no name."}), 400

    file_bytes = file.read()

    try:
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file_bytes)
        else:
            return jsonify({
                "error": "Unsupported file type. Please upload a PDF, PNG, or JPG."
            }), 400
    except Exception:
        return jsonify({
            "error": "Could not read this file. It may be corrupted, password-protected, or an unsupported format."
        }), 500

    if not text or not text.strip():
        return jsonify({
            "error": "No readable text was found in this file. For scanned images, make sure the photo is clear and well-lit."
        }), 400

    if len(text) > 20000:
        text = text[:20000]

    result = build_analysis(text)
    result["extracted_text"] = text
    result["source_filename"] = file.filename
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
