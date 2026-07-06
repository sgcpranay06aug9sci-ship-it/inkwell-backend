"""
app.py — Inkwell backend
-------------------------
A self-contained NLP engine (no external API keys, no heavy model downloads).
Implements:
  1. Sentence-level sentiment analysis (lexicon-based)
  2. Extractive summarization (word-frequency scoring)
  3. Keyword extraction (frequency-based, stopwords removed)
  4. Basic text statistics (word count, reading time)

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import re
from collections import Counter
from flask import Flask, request, jsonify

app = Flask(__name__)


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
    """Split text into sentences using punctuation boundaries."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text):
    """Lowercase word tokens, punctuation stripped."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_sentiment(sentence):
    """Return (label, score) for a single sentence using lexicon lookup."""
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
    """Aggregate sentence-level sentiment into one overall verdict."""
    if not sentences_data:
        return {"label": "neutral", "positive": 0, "negative": 0, "neutral": 0}
    counts = Counter(s["label"] for s in sentences_data)
    total = len(sentences_data)
    # Majority vote, ties go to neutral
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
    """Return the most frequent meaningful words (stopwords removed)."""
    words = tokenize(text)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def summarize(text, sentences, num_sentences=3):
    """Extractive summary: score sentences by frequency of their meaningful words."""
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
        # Slight boost for earlier sentences (often carry topic sentences)
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
    reading_time_min = max(1, round(word_count / 200))  # avg reading speed
    return {
        "words": word_count,
        "characters": char_count,
        "sentences": len(sentences),
        "reading_time": reading_time_min,
    }


# ---------------------------------------------------------------
# 3. ROUTES
# ---------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({"status": "Inkwell backend is running", "endpoint": "/api/analyze (POST)"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if len(text) > 20000:
        return jsonify({"error": "Text is too long. Please limit to 20,000 characters."}), 400

    sentences = split_sentences(text)

    sentences_data = []
    for s in sentences:
        label, score = sentence_sentiment(s)
        sentences_data.append({"text": s, "label": label, "score": score})

    result = {
        "sentiment": overall_sentiment(sentences_data),
        "sentences": sentences_data,
        "keywords": extract_keywords(text),
        "summary": summarize(text, sentences),
        "stats": text_stats(text, sentences),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
