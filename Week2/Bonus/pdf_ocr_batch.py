import os
import json
import requests
from pdf2image import convert_from_path
import pytesseract

# Optional: specify tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup folders
os.makedirs("pdfs", exist_ok=True)
os.makedirs("pdf_ocr", exist_ok=True)

# Load paper metadata from Task 1
with open("arxiv_clean.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Process each paper
for paper in papers:
    paper_id = paper["url"].split("/")[-1]
    pdf_path = f"pdfs/{paper_id}.pdf"
    txt_path = f"pdf_ocr/{paper_id}.txt"

    # Skip if already done
    if os.path.exists(txt_path):
        continue

    # Download PDF
    # Use category prefix for old IDs if needed
    prefix = "cs" if len(paper_id) <= 10 else ""
    pdf_url = f"https://arxiv.org/pdf/{prefix + '/' if prefix else ''}{paper_id}.pdf"

    response = requests.get(pdf_url)
    if response.status_code != 200:
        print(f"[!] Failed to download {pdf_url}")
        continue

    with open(pdf_path, "wb") as f:
        f.write(response.content)
    print(f"[+] Downloaded {pdf_path}")

    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"[!] PDF conversion error for {paper_id}: {e}")
        continue

    # OCR all pages and concatenate
    full_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        full_text += f"\n--- Page {i + 1} ---\n{text}"

    # Save to text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[✓] OCR saved to {txt_path}")
