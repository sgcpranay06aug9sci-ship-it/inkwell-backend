"""
app.py — Inkwell backend (v4: adds large-PDF support, up to ~1000 pages)
----------------------------------------------------------------------
Existing features:
  1. Sentence-level sentiment analysis (lexicon-based)
  2. Extractive summarization (word-frequency scoring)
  3. Keyword extraction (frequency-based, stopwords removed)
  4. Basic text statistics (word count, reading time)
  5. ML sentiment (trained Naive Bayes), readability score, word cloud data

New in this version:
  6. PDF text extraction is no longer truncated at 20,000 characters — the
     whole document is used.
  7. Large PDFs are summarized with a page-aware "map-reduce" extractive
     method: the doc is split into chunks, each chunk's best sentences are
     picked using that chunk's own word frequencies (so late pages aren't
     drowned out by early ones), then the candidates are re-ranked globally
     for the final summary. Every summary point is tagged with its page range.
  8. Large PDFs (more than SYNC_PAGE_LIMIT pages) are processed in a
     background thread and return a job_id right away (202 Accepted),
     instead of blocking the request until it times out. Poll
     GET /api/job-status/<job_id> for progress and the final result.
  9. Small PDFs, images, and pasted text still return an instant synchronous
     response, exactly as before — no changes needed on clients that don't
     poll.
"""

import re
import io
import time
import uuid
import threading
from collections import Counter
from flask import Flask, request, jsonify

from pypdf import PdfReader
from PIL import Image
import pytesseract

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

app = Flask(__name__)
# Raised from 15MB -> 75MB so large (mostly-text) PDFs aren't rejected outright.
# A 1000-page text PDF is usually 5-30MB. Tune this to your Render plan's
# memory ceiling if you see out-of-memory errors on very large files.
app.config["MAX_CONTENT_LENGTH"] = 75 * 1024 * 1024

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


def representative_sample(text, max_chars=20000):
    """For very long documents, take evenly spaced slices across the WHOLE
    text rather than just the beginning, so aggregate stats (ML sentiment,
    keywords, word cloud) reflect the full document instead of only its
    opening pages. Short text is returned unchanged."""
    if len(text) <= max_chars:
        return text
    num_slices = 10
    slice_size = max_chars // num_slices
    step = len(text) // num_slices
    parts = [text[i * step: i * step + slice_size] for i in range(num_slices)]
    return "\n".join(parts)


def build_analysis(text, pages_text=None, target_summary_sentences=12, sentence_display_limit=400):
    """
    Shared analysis pipeline used by the text endpoint and the file endpoint.

    If `pages_text` (a list of per-page extracted text) is supplied, summarization
    uses the page-aware hierarchical map-reduce method in hierarchical_summarize(),
    which scales to very long documents and tags each summary point with a page
    range. Otherwise the original single-pass word-frequency summarizer is used
    (short PDFs, images, and pasted text) — unchanged behavior.
    """
    sentences = split_sentences(text)
    words = tokenize(text)

    sentences_data = []
    for s in sentences:
        label, score = sentence_sentiment(s)
        sentences_data.append({"text": s, "label": label, "score": score})

    overall = overall_sentiment(sentences_data)  # computed on the FULL sentence set, never sampled

    # A 1000-page doc can have tens of thousands of sentences. Sending them all
    # to a mobile client is wasted payload since the UI only ever shows a
    # handful — ship an evenly-spaced sample instead, and report the true total.
    if len(sentences_data) > sentence_display_limit:
        step = len(sentences_data) / sentence_display_limit
        sentences_display = [sentences_data[int(i * step)] for i in range(sentence_display_limit)]
    else:
        sentences_display = sentences_data

    sample_text = representative_sample(text)

    if pages_text is not None:
        summary_result = hierarchical_summarize(pages_text, target_sentences=target_summary_sentences)
        summary_detailed = summary_result["summary"]
        summary_sentences = [item["text"] for item in summary_detailed]
    else:
        summary_sentences = summarize(text, sentences)
        summary_detailed = [{"text": s, "pages": None} for s in summary_sentences]

    return {
        "sentiment": overall,
        "ml_sentiment": ml_sentiment(sample_text),
        "sentences": sentences_display,
        "sentences_total": len(sentences_data),
        "keywords": extract_keywords(sample_text),
        "word_cloud": word_frequencies(sample_text),
        "summary": summary_sentences,
        "summary_detailed": summary_detailed,
        "stats": text_stats(text, sentences),
        "readability": readability_score(text, words, sentences),
    }


# ---------------------------------------------------------------
# 3. LONG-DOCUMENT SUMMARIZATION (map-reduce over pages, scales to ~1000 pages)
# ---------------------------------------------------------------

WORDS_PER_CHUNK = 2500   # roughly 6-8 pages per chunk depending on text density
SYNC_PAGE_LIMIT = 8      # PDFs at/under this many pages use the old instant, synchronous path


def build_word_freq(text):
    words = tokenize(text)
    filtered = [w for w in words if w not in STOPWORDS]
    freq = Counter(filtered)
    max_freq = max(freq.values()) if freq else 1
    return {w: c / max_freq for w, c in freq.items()}


def score_sentence(sentence, freq):
    words = tokenize(sentence)
    if not words:
        return 0.0
    # Mild length dampener so scoring doesn't just reward the longest sentence.
    return sum(freq.get(w, 0) for w in words) / (len(words) ** 0.3)


def chunk_pages(pages_text, words_per_chunk=WORDS_PER_CHUNK):
    """Group consecutive pages into ~words_per_chunk-word chunks, remembering
    which page range each chunk covers."""
    chunks = []
    current_text, current_word_count, start_page = [], 0, 1

    for i, page_text in enumerate(pages_text, start=1):
        current_text.append(page_text)
        current_word_count += len(tokenize(page_text))

        if current_word_count >= words_per_chunk or i == len(pages_text):
            chunks.append({
                "text": "\n".join(current_text),
                "start_page": start_page,
                "end_page": i,
            })
            current_text, current_word_count, start_page = [], 0, i + 1

    return [c for c in chunks if c["text"].strip()]


def hierarchical_summarize(pages_text, target_sentences=12):
    """
    Map-reduce extractive summarization for long documents:
      1. Split the document into page-bounded chunks (~2,500 words each).
      2. MAP: score every sentence in a chunk against THAT CHUNK's own word
         frequencies and keep its best few — so a document's later pages get
         a fair shot instead of being drowned out by whichever pages happen
         to share the most common words.
      3. REDUCE: re-score the surviving candidates against the document's
         GLOBAL word frequencies and keep the overall best `target_sentences`,
         restored to original document order.
    Every kept sentence is tagged with the page range it came from.
    """
    full_text = "\n".join(pages_text)
    chunks = chunk_pages(pages_text)
    if not chunks:
        return {"summary": [], "chunk_count": 0}

    global_freq = build_word_freq(full_text)

    # --- MAP step: pick local candidates from every chunk ---
    # Selection is by SENTENCE INDEX, not sentence text — a document with
    # repeated boilerplate (headers, disclaimers, recurring phrases) would
    # otherwise have every occurrence of a matched sentence text swept in,
    # letting one repeated line flood the final summary.
    per_chunk_keep = max(2, (target_sentences // max(1, len(chunks))) + 1)
    candidates = []  # (original_order, sentence_text, page_range_label)
    order = 0
    for chunk in chunks:
        local_freq = build_word_freq(chunk["text"])
        sentences = split_sentences(chunk["text"])
        scored = [(i, s, score_sentence(s, local_freq)) for i, s in enumerate(sentences)]
        top_indices = {i for i, s, sc in sorted(scored, key=lambda x: x[2], reverse=True)[:per_chunk_keep]}

        page_range = (f"p.{chunk['start_page']}" if chunk["start_page"] == chunk["end_page"]
                      else f"p.{chunk['start_page']}-{chunk['end_page']}")

        for i, s in enumerate(sentences):
            if i in top_indices:
                candidates.append((order, s, page_range))
                order += 1

    # --- REDUCE step: re-rank all candidates document-wide, deduping any
    # sentence text that recurred across multiple chunks (keep its best-scoring
    # occurrence's page reference; drop the rest so the summary isn't repetitive) ---
    rescored = sorted(
        [(idx, s, pr, score_sentence(s, global_freq)) for idx, s, pr in candidates],
        key=lambda x: x[3], reverse=True,
    )
    best, seen_text = [], set()
    for item in rescored:
        if item[1] in seen_text:
            continue
        seen_text.add(item[1])
        best.append(item)
        if len(best) >= target_sentences:
            break
    best_in_order = sorted(best, key=lambda x: x[0])

    return {
        "summary": [{"text": s, "pages": pr} for _, s, pr, _ in best_in_order],
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------
# 4. FILE TEXT EXTRACTION
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
# 5. BACKGROUND JOB QUEUE (in-memory — fine for a single-worker deployment)
# ---------------------------------------------------------------
# This dict lives in one process's memory. If you ever deploy with multiple
# gunicorn workers (e.g. `gunicorn -w 4`), each worker has its own copy and a
# client could poll a worker that never received their job. Keep this on a
# single worker (Render's default), or swap this for Redis if you scale up.

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60  # purge finished jobs an hour after creation


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if v.get("created_at", 0) < cutoff]:
            del JOBS[jid]


def _run_pdf_job(job_id, file_bytes, source_filename):
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "extracting"

        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        pages_text = []

        for i, page in enumerate(reader.pages):
            pages_text.append(page.extract_text() or "")
            if i % 20 == 0 or i == num_pages - 1:
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = {
                        "stage": "extracting", "pages_done": i + 1, "pages_total": num_pages,
                    }

        full_text = "\n".join(pages_text).strip()

        if not full_text:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = (
                    "No readable text was found in this PDF. It may be scanned "
                    "images rather than real text — multi-page OCR isn't enabled "
                    "in this version."
                )
            return

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "summarizing"
            JOBS[job_id]["progress"] = {"stage": "summarizing", "pages_total": num_pages}

        result = build_analysis(full_text, pages_text=pages_text)
        result["source_filename"] = source_filename
        result["page_count"] = num_pages

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"Failed to process PDF: {e}"


# ---------------------------------------------------------------
# 6. ROUTES
# ---------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "Inkwell backend is running",
        "endpoints": {
            "/api/analyze": "POST { text } -> analysis",
            "/api/analyze-file": "POST multipart file (pdf/png/jpg) -> analysis, "
                                  "or { job_id } (202) for PDFs over 8 pages",
            "/api/job-status/<job_id>": "GET -> { status, progress, result? } for large-PDF jobs",
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

    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
        except Exception:
            return jsonify({
                "error": "Could not read this PDF. It may be corrupted or password-protected."
            }), 500

        if num_pages > SYNC_PAGE_LIMIT:
            # Large PDF (this is what makes 1000-page documents possible): hand it
            # off to a background thread and return a job id immediately instead
            # of blocking the request until it times out.
            _purge_old_jobs()
            job_id = str(uuid.uuid4())
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "queued",
                    "progress": {"stage": "queued", "pages_total": num_pages},
                    "result": None,
                    "error": None,
                    "created_at": time.time(),
                }
            threading.Thread(target=_run_pdf_job, args=(job_id, file_bytes, file.filename), daemon=True).start()
            return jsonify({
                "job_id": job_id,
                "status": "queued",
                "page_count": num_pages,
                "message": "Large PDF received — poll /api/job-status/<job_id> for progress.",
            }), 202

        # Small PDF (8 pages or fewer): process synchronously, same as before.
        try:
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception:
            return jsonify({
                "error": "Could not read this file. It may be corrupted, password-protected, or an unsupported format."
            }), 500

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        try:
            text = extract_text_from_image(file_bytes)
        except Exception:
            return jsonify({
                "error": "Could not read this file. It may be corrupted, password-protected, or an unsupported format."
            }), 500
    else:
        return jsonify({
            "error": "Unsupported file type. Please upload a PDF, PNG, or JPG."
        }), 400

    if not text or not text.strip():
        return jsonify({
            "error": "No readable text was found in this file. For scanned images, make sure the photo is clear and well-lit."
        }), 400

    result = build_analysis(text)
    result["extracted_text"] = text
    result["source_filename"] = file.filename
    return jsonify(result)


@app.route("/api/job-status/<job_id>", methods=["GET"])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job id (it may have expired)."}), 404
        response = {"status": job["status"], "progress": job.get("progress")}
        if job["status"] == "done":
            response["result"] = job["result"]
        elif job["status"] == "error":
            response["error"] = job["error"]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
