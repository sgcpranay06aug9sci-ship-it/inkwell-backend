"""
app.py — Inkwell backend (v4: handles documents up to ~1000 pages)
---------------------------------------------------------------------
File uploads are processed as BACKGROUND JOBS, not single synchronous
requests, so large PDFs never hit a request timeout.

How large documents are handled:
  1. PDF pages are cleaned: page numbers and running headers/footers
     (lines repeated on many pages) are removed.
  2. The text is streamed in ~6-page chunks (never one giant word list).
  3. Each chunk gets a mini-summary; the mini-summaries are then combined
     and summarized once more into a final summary ("map-reduce").
  4. Keyword counts, sentiment and readability are accumulated chunk by chunk.
  5. Short inputs (one chunk) keep the original behaviour: every sentence
     is annotated and a 3-sentence summary is produced.
"""

import re
import io
import uuid
import threading
import time
from collections import Counter
from flask import Flask, request, jsonify

from pypdf import PdfReader
from PIL import Image
import pytesseract

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB upload limit


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------
# ML SENTIMENT MODEL (trained once at server startup)
# ---------------------------------------------------------------
print("Training ML sentiment model (one-time, ~5 seconds)...")
ML_ANALYZER = NaiveBayesAnalyzer()
TextBlob("warming up the model", analyzer=ML_ANALYZER).sentiment
print("ML sentiment model ready.")


def ml_sentiment(text):
    result = TextBlob(text, analyzer=ML_ANALYZER).sentiment
    confidence = round(max(result.p_pos, result.p_neg) * 100)
    label = "positive" if result.classification == "pos" else "negative"
    return {"label": label, "confidence": confidence}


# ---------------------------------------------------------------
# STATIC RESOURCES
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
# CORE TEXT HELPERS
# ---------------------------------------------------------------

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_sentiment(sentence):
    words = tokenize(sentence)
    score = sum(1 for w in words if w in POSITIVE_WORDS) - sum(1 for w in words if w in NEGATIVE_WORDS)
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score


def overall_sentiment_from_labels(labels):
    if not labels:
        return {"label": "neutral", "positive": 0, "negative": 0, "neutral": 0}
    counts = Counter(labels)
    total = len(labels)
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


def count_syllables(word):
    word = word.lower()
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_from_counts(word_count, sentence_count, syllable_count):
    if word_count == 0 or sentence_count == 0:
        return {"grade_level": 0, "label": "N/A"}
    grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
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


def clean_candidates(sentences, seen):
    """Drop tiny fragments and duplicate sentences. `seen` (a set of hashes) is
    shared across the whole document so repeats on later pages are caught."""
    kept = []
    for s in sentences:
        if len(tokenize(s)) < 6:
            continue
        key = hash(re.sub(r"\W+", " ", s.lower()).strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    return kept


def rank_sentences(sentences, freq, top_n, boost_first=False):
    """Pick the top_n highest-scoring sentences, keeping their original order."""
    if len(sentences) <= top_n:
        return sentences
    max_freq = max(freq.values()) if freq else 1
    normalized = {w: c / max_freq for w, c in freq.items()}
    scored = []
    for idx, s in enumerate(sentences):
        score = sum(normalized.get(w, 0) for w in tokenize(s))
        if boost_first and idx == 0:
            score *= 1.2
        scored.append((idx, s, score))
    top = sorted(scored, key=lambda x: x[2], reverse=True)[:top_n]
    return [s for _, s, _ in sorted(top, key=lambda x: x[0])]


# ---------------------------------------------------------------
# JOB STORE (in-memory — requires a single gunicorn worker process)
# ---------------------------------------------------------------

JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS[job_id]) if job_id in JOBS else None


# ---------------------------------------------------------------
# FILE TEXT EXTRACTION
# ---------------------------------------------------------------

PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)


def normalize_line(line):
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pdf_pages(pages):
    """Remove page numbers and any line repeated on many pages (running
    headers/footers); end short heading-like lines with a period so they don't
    get glued onto the next real sentence."""
    split_pages = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]

    repeated = set()
    if len(pages) >= 8:
        counts = Counter()
        for lines in split_pages:
            counts.update({normalize_line(l) for l in lines})
        threshold = max(4, int(len(pages) * 0.25))
        repeated = {k for k, v in counts.items() if v >= threshold}

    cleaned = []
    for lines in split_pages:
        kept = []
        for l in lines:
            if PAGE_NUM_RE.match(l) or normalize_line(l) in repeated:
                continue
            if len(l) < 50 and len(l.split()) < 8 and l[-1] not in ".!?:;,":
                l += "."
            kept.append(l)
        cleaned.append(" ".join(kept))
    return "\n".join(cleaned)


def extract_text_from_pdf(file_bytes, job_id=None):
    reader = PdfReader(io.BytesIO(file_bytes))
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append(page.extract_text() or "")
        if job_id and (i % 20 == 0 or i == total - 1):
            set_job(job_id, progress=f"Extracting text — page {i + 1} of {total}")
    if job_id:
        set_job(job_id, progress="Cleaning headers, footers and page numbers")
    return clean_pdf_pages(pages).strip(), total


def extract_text_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return pytesseract.image_to_string(image).strip()


# ---------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------

CHUNK_CHAR_SIZE = 18000          # ~3000 words (~6 pages) per chunk
PER_CHUNK_SUMMARY_SENTENCES = 2  # mini-summary size per chunk
FINAL_SUMMARY_SENTENCES = 7      # size of the final combined summary


def iter_chunks(text, size=CHUNK_CHAR_SIZE):
    """Yield the text in ~`size`-character slices, always cutting at a space.
    Streaming like this means we never hold a list of every word in the document."""
    n, i = len(text), 0
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.rfind(" ", i, j)
            if k > i:
                j = k
        yield text[i:j]
        i = j


def analyze_small_text(text):
    """Short inputs (one chunk): annotate EVERY sentence, 3-sentence summary."""
    sentences = split_sentences(text)
    words = tokenize(text)
    freq = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)

    sentences_data = []
    for s in sentences:
        label, score = sentence_sentiment(s)
        sentences_data.append({"text": s, "label": label, "score": score})

    syllables = sum(count_syllables(w) for w in words)
    word_total = len(text.split())
    return {
        "sentiment": overall_sentiment_from_labels([d["label"] for d in sentences_data]),
        "ml_sentiment": ml_sentiment(text),
        "sentences": sentences_data,
        "keywords": [w for w, _ in freq.most_common(8)],
        "word_cloud": [{"word": w, "count": c} for w, c in freq.most_common(25)],
        "summary": rank_sentences(sentences, freq, 3, boost_first=True),
        "stats": {
            "words": word_total,
            "characters": len(text),
            "sentences": len(sentences),
            "reading_time": max(1, round(word_total / 200)),
        },
        "readability": readability_from_counts(len(words), max(1, len(sentences)), syllables),
        "chunks_processed": 1,
    }


def analyze_large_text(text, job_id=None):
    if len(text) <= CHUNK_CHAR_SIZE:
        return analyze_small_text(text)

    num_chunks = -(-len(text) // CHUNK_CHAR_SIZE)  # ceiling division
    global_word_freq = Counter()
    chunk_sentiment_labels = []
    candidate_summary_sentences = []
    seen_hashes = set()
    total_word_count = total_sentence_count = total_syllable_count = 0

    for c, chunk_text in enumerate(iter_chunks(text)):
        if not chunk_text.strip():
            continue
        if job_id and (c % 5 == 0 or c == num_chunks - 1):
            set_job(job_id, progress=f"Analyzing section {c + 1} of {num_chunks}")

        total_word_count += len(chunk_text.split())
        chunk_sentences = split_sentences(chunk_text)
        total_sentence_count += len(chunk_sentences)

        chunk_words = tokenize(chunk_text)
        filtered = [w for w in chunk_words if w not in STOPWORDS and len(w) > 2]
        global_word_freq.update(filtered)
        total_syllable_count += sum(count_syllables(w) for w in chunk_words)

        pos = sum(1 for w in chunk_words if w in POSITIVE_WORDS)
        neg = sum(1 for w in chunk_words if w in NEGATIVE_WORDS)
        chunk_sentiment_labels.append("positive" if pos > neg else "negative" if neg > pos else "neutral")

        usable = clean_candidates(chunk_sentences, seen_hashes)
        if usable:
            candidate_summary_sentences.extend(
                rank_sentences(usable, Counter(filtered), PER_CHUNK_SUMMARY_SENTENCES, boost_first=(c == 0))
            )

    if job_id:
        set_job(job_id, progress="Combining section summaries into a final summary")

    final_summary = rank_sentences(candidate_summary_sentences, global_word_freq, FINAL_SUMMARY_SENTENCES)
    if not final_summary:
        final_summary = split_sentences(text)[:FINAL_SUMMARY_SENTENCES]

    summary_data = []
    for s in final_summary:
        label, score = sentence_sentiment(s)
        summary_data.append({"text": s, "label": label, "score": score})

    if job_id:
        set_job(job_id, progress="Running ML sentiment on summary")
    ml_result = ml_sentiment(" ".join(final_summary)) if final_summary else {"label": "neutral", "confidence": 0}

    return {
        "sentiment": overall_sentiment_from_labels(chunk_sentiment_labels),
        "ml_sentiment": ml_result,
        "sentences": summary_data,   # for big documents the annotated view shows the key sentences
        "keywords": [w for w, _ in global_word_freq.most_common(8)],
        "word_cloud": [{"word": w, "count": c} for w, c in global_word_freq.most_common(25)],
        "summary": final_summary,
        "stats": {
            "words": total_word_count,
            "characters": len(text),
            "sentences": total_sentence_count,
            "reading_time": max(1, round(total_word_count / 200)),
        },
        "readability": readability_from_counts(total_word_count, max(1, total_sentence_count), total_syllable_count),
        "chunks_processed": num_chunks,
    }


# ---------------------------------------------------------------
# BACKGROUND JOB WORKER
# ---------------------------------------------------------------

def run_file_job(job_id, file_bytes, filename):
    try:
        lower = (filename or "").lower()
        set_job(job_id, status="processing", progress="Starting…")

        if lower.endswith(".pdf"):
            text, page_count = extract_text_from_pdf(file_bytes, job_id=job_id)
        elif lower.endswith((".png", ".jpg", ".jpeg")):
            set_job(job_id, progress="Running OCR on photo…")
            text = extract_text_from_image(file_bytes)
            page_count = 1
        else:
            set_job(job_id, status="error", error="Unsupported file type. Please upload a PDF, PNG, or JPG.")
            return

        if not text or not text.strip():
            set_job(job_id, status="error",
                    error="No readable text was found in this file. Scanned PDFs (pictures of pages) are not supported — upload a text PDF, or photograph a single page instead.")
            return

        result = analyze_large_text(text, job_id=job_id)
        result["source_filename"] = filename
        result["page_count"] = page_count
        result["extracted_text_preview"] = text[:4000]
        set_job(job_id, status="done", progress="Complete", result=result)

    except Exception as e:
        set_job(job_id, status="error", error=f"Something went wrong while processing this file: {e}")


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "Inkwell backend is running",
        "endpoints": {
            "/api/analyze": "POST { text } -> analysis",
            "/api/analyze-file": "POST multipart file (pdf/png/jpg) -> { job_id }",
            "/api/job/<job_id>": "GET -> job status / result",
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 2_000_000:
        return jsonify({"error": "Pasted text is too long. Please upload it as a PDF instead."}), 400
    return jsonify(analyze_large_text(text))


@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Uploaded file has no name."}), 400

    file_bytes = file.read()
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "pending", "progress": "Queued…", "result": None,
                        "error": None, "created": time.time()}

    threading.Thread(target=run_file_job, args=(job_id, file_bytes, file.filename), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found. It may have expired."}), 404
    response = {"status": job["status"], "progress": job.get("progress", "")}
    if job["status"] == "done":
        response["result"] = job["result"]
    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error.")
    return jsonify(response)


def cleanup_old_jobs():
    while True:
        time.sleep(600)
        cutoff = time.time() - 3600
        with JOBS_LOCK:
            for jid in [j for j, v in JOBS.items() if v.get("created", 0) < cutoff]:
                del JOBS[jid]


threading.Thread(target=cleanup_old_jobs, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001, threaded=True)
