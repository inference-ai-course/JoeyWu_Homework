# arXiv_scraper.py
# Query arXiv API for cs.AI papers and download PDFs into data/arxiv

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# -------------------- Settings --------------------
CATEGORY    = "cs.AI"     # <-- AI subcategory (change to "cs.CL" etc. if needed)
MAX_PAPERS  = 50          # how many PDFs to download total
PER_PAGE    = 100         # API page size (<= 200 is safe)
SLEEP_S     = 0.5         # politeness delay between downloads
TIMEOUT     = 60
BASE        = "https://arxiv.org"

# Output dir is relative to this script
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "data" / "arxiv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "arxiv-api-downloader (+https://github.com/your-handle)"}

# -------------------- HTTP session w/ retries --------------------
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(HEADERS)
    return sess

# -------------------- Helpers --------------------
def build_api_url(start: int, max_results: int) -> str:
    # Sort newest first
    return (
        "http://export.arxiv.org/api/query"
        f"?search_query=cat:{CATEGORY}"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&start={start}&max_results={max_results}"
    )

def sanitize(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[^\w\-\.\s]+", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:max_len].rstrip(" ._-")

def parse_api_feed(xml_text: str) -> list[dict]:
    """
    Return items like:
    {'id': '2508.10759v1', 'title': '...', 'pdf_url': 'https://arxiv.org/pdf/2508.10759v1'}
    """
    soup = BeautifulSoup(xml_text, "xml")
    items = []
    for entry in soup.find_all("entry"):
        arxiv_id = entry.id.text.rsplit("/", 1)[-1]  # e.g., 2508.10759v1
        title = entry.title.text.strip()

        # Prefer explicit pdf link when present
        link_pdf = entry.find("link", title="pdf")
        pdf_url = link_pdf["href"] if link_pdf and link_pdf.has_attr("href") else f"{BASE}/pdf/{arxiv_id}"

        items.append({"id": arxiv_id, "title": title, "pdf_url": pdf_url})
    return items

def download_pdf(sess: requests.Session, item: dict, out_dir: Path) -> Path:
    filename = f"{item['id']} - {sanitize(item['title']) or item['id']}.pdf"
    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"✓ Exists, skipping: {out_path.name}")
        return out_path

    with sess.get(item["pdf_url"], stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"⬇️  Downloaded: {out_path.name}")
    return out_path

# -------------------- Main --------------------
def main():
    sess = make_session()

    # Page through API until we have MAX_PAPERS (or run out)
    items, seen = [], set()
    start = 0
    while len(items) < MAX_PAPERS:
        url = build_api_url(start, min(PER_PAGE, MAX_PAPERS - len(items)))
        r = sess.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        page_items = parse_api_feed(r.text)
        if not page_items:
            break
        for it in page_items:
            if it["id"] not in seen:
                items.append(it)
                seen.add(it["id"])
                if len(items) >= MAX_PAPERS:
                    break
        start += len(page_items)

    if not items:
        print("✗ No items returned by the API.")
        return

    print(f"Will download {len(items)} PDFs into: {OUT_DIR.resolve()}")
    for i, it in enumerate(items, 1):
        print(f"[{i:03d}/{len(items)}] {it['id']} — {it['title'][:80]}")
        try:
            download_pdf(sess, it, OUT_DIR)
        except Exception as e:
            print(f"   ✗ Failed: {e}")
        time.sleep(SLEEP_S)

    print(f"\n✅ Done. PDFs saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
