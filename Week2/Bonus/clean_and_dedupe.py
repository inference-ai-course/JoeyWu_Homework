import os
import json
import glob
import re
import hashlib
import utils
from langdetect import detect
from datasketch import MinHash, MinHashLSH

TASK_1_OUTPUT = "arxiv_clean.json"
TASK_2_OUTPUT = "pdfs/*.txt"
TASK_3_OUTPUT = "../talks_transcripts/talks_transcripts.jsonl"
MERGED_OUTPUT = "clean_corpus.txt"
STATS_OUTPUT = "stats.md"

MINHASH_THRESHOLD = 0.7

# --------------- Text Extraction ---------------
def extract_text():
    all_texts = []

    task_1_path = utils.get_path(TASK_1_OUTPUT)
    if os.path.exists(task_1_path):
        with open(task_1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_texts.extend([entry.get("abstract", "").strip() for entry in data if entry.get("abstract")])

    for file_path in glob.glob(utils.get_path(TASK_2_OUTPUT)):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_texts.append(content)
        except Exception:
            pass

    task_3_path = utils.get_path(TASK_3_OUTPUT)
    if os.path.exists(task_3_path):
        with open(task_3_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "").strip()
                    if text:
                        all_texts.append(text)
                except json.JSONDecodeError:
                    continue

    return all_texts

# --------------- Cleaning Functions ---------------
def strip_html(text):
    return re.sub(r"<[^>]+>", "", text)

def remove_pii(text):
    text = re.sub(r"[\w.-]+@[\w.-]+", "", text)  # emails
    text = re.sub(r"\b\d{13,19}\b", "", text)    # credit card
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", text)  # phone numbers
    return text

def remove_repetitive_ngrams(text, n=5):
    tokens = text.split()
    seen = set()
    result = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        if ngram not in seen:
            seen.add(ngram)
            result.append(tokens[i])
    result.extend(tokens[len(result):])
    return " ".join(result)

# --------------- Deduplication via MinHash ---------------
def is_similar(a, b, threshold=MINHASH_THRESHOLD):
    m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
    for token in a.split(): m1.update(token.encode("utf8"))
    for token in b.split(): m2.update(token.encode("utf8"))
    return m1.jaccard(m2) >= threshold

def deduplicate_texts(texts):
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=128)
    unique = []
    hash_to_text = {}
    for idx, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for token in text.split():
            m.update(token.encode("utf8"))
        key = f"doc_{idx}"
        if not any(lsh.query(m)):
            lsh.insert(key, m)
            unique.append(text)
            hash_to_text[key] = text
    return unique

# --------------- Processing Pipeline ---------------
def process_clean_and_dedupe():
    raw_texts = extract_text()
    initial_count = len(raw_texts)

    cleaned = []
    for t in raw_texts:
        try:
            if detect(t) != "en":
                continue
        except:
            continue
        t = strip_html(t)
        t = remove_pii(t)
        t = remove_repetitive_ngrams(t)
        cleaned.append(t.strip())

    deduped = deduplicate_texts(cleaned)

    with open(utils.get_path(MERGED_OUTPUT), "w", encoding="utf-8") as f:
        for line in deduped:
            f.write(line + "\n")

    with open(utils.get_path(STATS_OUTPUT), "w", encoding="utf-8") as f:
        f.write(f"Original texts: {initial_count}\n")
        f.write(f"After cleaning & deduplication: {len(deduped)}\n")
        f.write(f"Removed: {initial_count - len(deduped)} ({(initial_count - len(deduped)) / initial_count:.2%})\n")

    print("[✓] Cleaned and deduplicated corpus written to", MERGED_OUTPUT)
    print("[✓] Stats written to", STATS_OUTPUT)

if __name__ == "__main__":
    process_clean_and_dedupe()
