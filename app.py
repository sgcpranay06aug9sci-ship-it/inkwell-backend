"""
<<<<<<< HEAD
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
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
"""

import re
import io
<<<<<<< HEAD
import uuid
import threading
import time
=======
import time
import uuid
import threading
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
from collections import Counter
from flask import Flask, request, jsonify

from pypdf import PdfReader
from PIL import Image
import pytesseract

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

app = Flask(__name__)
<<<<<<< HEAD
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB upload limit
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc


@app.after_request
def add_cors_headers(response):
<<<<<<< HEAD
=======
    """Allow the mobile app (running on a phone, a different origin) to call this API."""
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------------------------------------------------------
<<<<<<< HEAD
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
=======
# 1. STATIC RESOURCES (no downloads needed — everything is local)
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
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
<<<<<<< HEAD
# CORE TEXT HELPERS
=======
# 2. TEXT PROCESSING HELPERS
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
# ---------------------------------------------------------------

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
<<<<<<< HEAD
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
=======
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_sentiment(sentence):
    words = tokenize(sentence)
<<<<<<< HEAD
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
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
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


<<<<<<< HEAD
def count_syllables(word):
    word = word.lower()
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in "aeiouy"
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


<<<<<<< HEAD
def readability_from_counts(word_count, sentence_count, syllable_count):
    if word_count == 0 or sentence_count == 0:
        return {"grade_level": 0, "label": "N/A"}
    grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
    grade = max(0, round(grade, 1))
=======
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

>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
    if grade <= 6:
        label = "Easy (elementary level)"
    elif grade <= 9:
        label = "Moderate (middle school level)"
    elif grade <= 13:
        label = "Difficult (high school / college level)"
    else:
        label = "Very difficult (graduate level)"
<<<<<<< HEAD
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
=======

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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc

    sentences_data = []
    for s in sentences:
        label, score = sentence_sentiment(s)
        sentences_data.append({"text": s, "label": label, "score": score})

<<<<<<< HEAD
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
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
    }


# ---------------------------------------------------------------
<<<<<<< HEAD
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
=======
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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
# ---------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "Inkwell backend is running",
        "endpoints": {
            "/api/analyze": "POST { text } -> analysis",
<<<<<<< HEAD
            "/api/analyze-file": "POST multipart file (pdf/png/jpg) -> { job_id }",
            "/api/job/<job_id>": "GET -> job status / result",
=======
            "/api/analyze-file": "POST multipart file (pdf/png/jpg) -> analysis, "
                                  "or { job_id } (202) for PDFs over 8 pages",
            "/api/job-status/<job_id>": "GET -> { status, progress, result? } for large-PDF jobs",
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
<<<<<<< HEAD
    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 2_000_000:
        return jsonify({"error": "Pasted text is too long. Please upload it as a PDF instead."}), 400
    return jsonify(analyze_large_text(text))
=======

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text is too long. Please limit to 20,000 characters."}), 400

    result = build_analysis(text)
    return jsonify(result)
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc


@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
<<<<<<< HEAD
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
=======

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
>>>>>>> d0b184891c7b3b7527c94953f8e4c552dc63a5dc
