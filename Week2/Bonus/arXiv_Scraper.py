import os
import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import pytesseract
import trafilatura

# Set subcategory and how many papers to fetch
CATEGORY = 'cs.CL'
MAX_RESULTS = 5
OUTPUT_FILE = 'arxiv_clean.json'

def fetch_arxiv_papers(category, max_results):
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch arXiv API")
    return response.text

def parse_arxiv_feed(feed):
    soup = BeautifulSoup(feed, 'xml')
    entries = soup.find_all('entry')
    papers = []
    for entry in entries:
        paper = {
            'url': entry.id.text,
            'title': entry.title.text.strip(),
            'abstract': entry.summary.text.strip(),
            'authors': [author.find('name').text for author in entry.find_all('author')],
            'date': entry.published.text[:10]
        }
        papers.append(paper)
    return papers

def extract_clean_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return None

def ocr_from_webpage_image(url):
    # Simulated OCR: take screenshot (manually or with headless browser), then run OCR.
    # Here we simulate this using a sample image, since rendering JS webpages needs selenium.
    print(f"[!] OCR step skipped (requires screenshot of {url})")
    return "Simulated OCR abstract text"

def save_to_json(papers, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def main():
    feed = fetch_arxiv_papers(CATEGORY, MAX_RESULTS)
    papers = parse_arxiv_feed(feed)

    for paper in papers:
        print(f"Processing: {paper['title']}")
        # Try extracting clean text using trafilatura
        clean_text = extract_clean_text_from_url(paper['url'])
        if clean_text:
            paper['abstract'] = clean_text
        else:
            # If failed, fallback to OCR
            paper['abstract'] = ocr_from_webpage_image(paper['url'])

    save_to_json(papers, OUTPUT_FILE)
    print(f"\n✅ Saved {len(papers)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
