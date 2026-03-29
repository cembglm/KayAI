"""
Eskişehir Kayaç Parkı - Veri Çekme Scripti
Bu script OGM sitesinden tüm 81 kayacın verilerini çeker ve data_cache klasörüne kaydeder.
Bir kez çalıştırılması yeterlidir.
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://www.ogm.gov.tr"
KAYAC_MAIN = "https://www.ogm.gov.tr/eskisehirobm/eskisehir_kayac_parki"
KAYAC_DETAIL_TEMPLATE = "https://www.ogm.gov.tr/eskisehirobm/Sayfalar/Eskisehir_Kayac_Parki/{:03d}.aspx"
DATA_DIR = "data_cache"
REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

os.makedirs(DATA_DIR, exist_ok=True)


def http_get(url):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


def clean_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def soup_text_blocks(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks = []
    main = soup.find("main") or soup.find(id="content") or soup.body
    if not main:
        return blocks
    for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t and len(t) >= 25:
            blocks.append(t)
    seen = set()
    uniq = []
    for b in blocks:
        key = b[:200]
        if key not in seen:
            uniq.append(b)
            seen.add(key)
    return uniq


def find_pdf_links(html):
    soup = BeautifulSoup(html, "html.parser")
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ".pdf" in href.lower():
            if href.startswith("/"):
                href = BASE_URL + href
            elif not href.startswith("http"):
                href = BASE_URL + "/" + href.lstrip("/")
            pdfs.append(href)
    return list(dict.fromkeys(pdfs))


def find_images(html):
    soup = BeautifulSoup(html, "html.parser")
    images = []
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if any(skip in src.lower() for skip in ["icon", "logo", "ataturk", "banner", "header", "footer", ".gif"]):
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = BASE_URL + src
        elif not src.startswith("http"):
            src = BASE_URL + "/" + src.lstrip("/")
        images.append(src)
    return list(dict.fromkeys(images))


def main():
    print("=" * 60)
    print("Eskişehir Kayaç Parkı - Veri Çekme İşlemi")
    print("=" * 60)
    print()

    rocks_data = {}
    rock_images = {}

    print("81 kayaç verisi çekiliyor...")
    print("-" * 40)

    for i in range(1, 82):
        try:
            url = KAYAC_DETAIL_TEMPLATE.format(i)
            r = http_get(url)
            soup = BeautifulSoup(r.text, "html.parser")
            h1 = soup.find("h1")
            title = clean_text(h1.get_text(" ", strip=True)) if h1 else f"Kayaç {i:03d}"
            blocks = soup_text_blocks(r.text)
            pdfs = find_pdf_links(r.text)
            images = find_images(r.text)

            doc_id = f"rock_{i:03d}"
            rocks_data[doc_id] = {
                "doc_id": doc_id,
                "title": title or f"Kayaç {i:03d}",
                "url": url,
                "text": "\n".join(blocks),
                "pdf_links": pdfs
            }
            if images:
                rock_images[doc_id] = {"images": images, "title": title}

            print(f"  [{i:03d}/081] ✓ {title}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{i:03d}/081] ✗ HATA: {e}")
            rocks_data[f"rock_{i:03d}"] = {
                "doc_id": f"rock_{i:03d}",
                "title": f"Kayaç {i:03d}",
                "url": KAYAC_DETAIL_TEMPLATE.format(i),
                "text": "",
                "pdf_links": [],
                "error": str(e)
            }

    # Save rocks data
    rocks_file = os.path.join(DATA_DIR, "rocks_data.json")
    with open(rocks_file, "w", encoding="utf-8") as f:
        json.dump(rocks_data, f, ensure_ascii=False, indent=2)

    # Save rock images
    images_file = os.path.join(DATA_DIR, "rock_images.json")
    with open(images_file, "w", encoding="utf-8") as f:
        json.dump(rock_images, f, ensure_ascii=False, indent=2)

    # Create documents.jsonl for search index
    print()
    print("Arama indeksi oluşturuluyor...")
    
    chunks = []
    for doc_id, data in rocks_data.items():
        text = data.get("text", "")
        if text:
            # Simple chunking
            chunk = {
                "doc_id": doc_id,
                "title": data.get("title", doc_id),
                "url": data.get("url", ""),
                "chunk_id": f"{doc_id}::c0",
                "text": text[:900] if len(text) > 900 else text
            }
            chunks.append(chunk)

    docs_file = os.path.join(DATA_DIR, "documents.jsonl")
    with open(docs_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Create meta.json
    meta = {
        "status": "ready",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_rocks": len(rocks_data),
        "num_chunks": len(chunks),
        "source_main": KAYAC_MAIN
    }
    meta_file = os.path.join(DATA_DIR, "meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print(f"  • Kayaç sayısı: {len(rocks_data)}")
    print(f"  • Görsel bulunan: {len(rock_images)}")
    print(f"  • Arama parçası: {len(chunks)}")
    print()
    print(f"Dosyalar kaydedildi:")
    print(f"  • {rocks_file}")
    print(f"  • {images_file}")
    print(f"  • {docs_file}")
    print(f"  • {meta_file}")


if __name__ == "__main__":
    main()
