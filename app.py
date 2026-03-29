import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF parsing (fallback chain)
PDF_TEXT_OK = True
try:
    import PyPDF2
except Exception:
    PDF_TEXT_OK = False

# Google Gemini AI
GEMINI_AVAILABLE = True
try:
    from google import genai
    from google.genai import types
except ImportError:
    GEMINI_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------
BASE_URL = "https://www.ogm.gov.tr"
KAYAC_MAIN = "https://www.ogm.gov.tr/eskisehirobm/eskisehir_kayac_parki"
KAYAC_DETAIL_TEMPLATE = "https://www.ogm.gov.tr/eskisehirobm/Sayfalar/Eskisehir_Kayac_Parki/{:03d}.aspx"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20

DATA_DIR = "data_cache"
DOCS_JSONL = os.path.join(DATA_DIR, "documents.jsonl")
META_JSON = os.path.join(DATA_DIR, "meta.json")
VECTORIZER_BIN = os.path.join(DATA_DIR, "vectorizer.pkl")
MATRIX_BIN = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
ROCKS_DATA_JSON = os.path.join(DATA_DIR, "rocks_data.json")

# Gemini AI Configuration
# API key is stored securely in .streamlit/secrets.toml
GEMINI_MODEL = "gemini-2.0-flash"
ENRICHED_ROCKS_JSON = os.path.join(DATA_DIR, "enriched_rocks.json")
ROCK_IMAGES_JSON = os.path.join(DATA_DIR, "rock_images.json")


def get_gemini_api_key() -> str:
    """Get Gemini API key from st.secrets (secure for deployment)."""
    try:
        return st.secrets["gemini"]["api_key"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    """Get Gemini model name from st.secrets or default."""
    try:
        return st.secrets["gemini"]["model"]
    except (KeyError, FileNotFoundError):
        return GEMINI_MODEL


# -----------------------------
# Utilities
# -----------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def http_get(url: str) -> requests.Response:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def soup_text_blocks(html: str) -> List[str]:
    """
    Extracts readable text blocks from the main content area.
    We keep it simple and robust: collect paragraph-ish blocks.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks = []
    # Prefer main content if exists
    main = soup.find("main") or soup.find(id="content") or soup.body
    if not main:
        return blocks

    # paragraphs, list items, headings
    for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
        t = clean_text(tag.get_text(" ", strip=True))
        if t and len(t) >= 25:
            blocks.append(t)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for b in blocks:
        key = b[:200]
        if key not in seen:
            uniq.append(b)
            seen.add(key)
    return uniq


def find_pdf_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ".pdf" in href.lower():
            if href.startswith("/"):
                href = BASE_URL + href
            elif href.startswith("http"):
                pass
            else:
                href = BASE_URL + "/" + href.lstrip("/")
            pdfs.append(href)
    # unique
    out = []
    seen = set()
    for p in pdfs:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not PDF_TEXT_OK:
        return ""
    try:
        reader = PyPDF2.PdfReader(io_bytes := __import__("io").BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            t = clean_text(t)
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""


def find_images(html: str, base_url: str = BASE_URL) -> List[str]:
    """Extract image URLs from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    images = []
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        # Skip tiny icons and common assets
        if any(skip in src.lower() for skip in ["icon", "logo", "ataturk", "banner", "header", "footer", ".gif"]):
            continue
        # Build absolute URL
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = base_url + src
        elif not src.startswith("http"):
            src = base_url + "/" + src.lstrip("/")
        images.append(src)
    # Deduplicate
    return list(dict.fromkeys(images))


@dataclass
class DocChunk:
    doc_id: str
    title: str
    url: str
    chunk_id: str
    text: str


def chunk_document(doc_id: str, title: str, url: str, full_text: str,
                   max_chars: int = 900, overlap: int = 120) -> List[DocChunk]:
    """
    Simple character-based chunking preserving sentence boundaries loosely.
    """
    full_text = full_text.strip()
    if not full_text:
        return []

    # Split by sentence-ish boundaries, then pack
    parts = re.split(r"(?<=[.!?])\s+", full_text)
    chunks = []
    buf = ""
    idx = 0

    def flush(textbuf: str):
        nonlocal idx
        t = clean_text(textbuf)
        if len(t) >= 80:
            chunks.append(
                DocChunk(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    chunk_id=f"{doc_id}::c{idx}",
                    text=t
                )
            )
            idx += 1

    for p in parts:
        p = clean_text(p)
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            flush(buf)
            # overlap: keep tail of previous
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + " " + p).strip()

    if buf:
        flush(buf)
    return chunks


# -----------------------------
# Scraping / Ingest
# -----------------------------
def scrape_main_page() -> Dict[str, Any]:
    r = http_get(KAYAC_MAIN)
    blocks = soup_text_blocks(r.text)
    pdfs = find_pdf_links(r.text)
    return {
        "title": "Eskişehir Kayaç Parkı (Ana Sayfa)",
        "url": KAYAC_MAIN,
        "text": "\n".join(blocks),
        "pdf_links": pdfs,
    }


def scrape_rock_page(no: int) -> Dict[str, Any]:
    url = KAYAC_DETAIL_TEMPLATE.format(no)
    r = http_get(url)

    # Try to infer rock name/title
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else f"Kayaç {no:03d}"

    blocks = soup_text_blocks(r.text)
    pdfs = find_pdf_links(r.text)
    images = find_images(r.text)

    return {
        "title": title or f"Kayaç {no:03d}",
        "url": url,
        "text": "\n".join(blocks),
        "pdf_links": pdfs,
        "images": images,
    }


def ingest_all(force: bool = False) -> Dict[str, Any]:
    """
    Fetch:
    - main page
    - 001..081 pages
    - first relevant PDF if available on main page (and/or rock pages)
    Build documents.jsonl with chunked passages.
    """
    ensure_data_dir()

    # meta cache
    meta = {}
    if os.path.exists(META_JSON):
        try:
            meta = json.load(open(META_JSON, "r", encoding="utf-8"))
        except Exception:
            meta = {}

    if (not force) and meta.get("status") == "ready" and os.path.exists(DOCS_JSONL):
        return {"ok": True, "message": "Önbellek zaten hazır. Zorla güncelle için 'force' kullanın."}

    docs_raw = []
    rock_images = {}  # Store images for each rock

    # Main page
    st.info("Ana sayfa çekiliyor...")
    main_doc = scrape_main_page()
    docs_raw.append({"doc_id": "main", **main_doc})
    time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    # Rock pages
    st.info("Kayaç sayfaları çekiliyor (001-081)...")
    progress_bar = st.progress(0)
    for i in range(1, 82):
        try:
            d = scrape_rock_page(i)
            docs_raw.append({"doc_id": f"rock_{i:03d}", **d})
            # Store images for this rock
            if d.get("images"):
                rock_images[f"rock_{i:03d}"] = {
                    "images": d["images"],
                    "title": d.get("title", f"Kayaç {i:03d}")
                }
        except Exception as e:
            docs_raw.append({"doc_id": f"rock_{i:03d}", "title": f"Kayaç {i:03d}", "url": KAYAC_DETAIL_TEMPLATE.format(i), "text": "", "pdf_links": [], "images": [], "error": str(e)})
        progress_bar.progress(i / 81)
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    progress_bar.empty()

    # Save rock images
    with open(ROCK_IMAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(rock_images, f, ensure_ascii=False, indent=2)

    # PDF: Try to get ALL PDFs from rock pages for richer content
    st.info("PDF'ler işleniyor...")
    pdf_count = 0
    all_pdfs = []
    for d in docs_raw:
        all_pdfs.extend(d.get("pdf_links", []))
    all_pdfs = list(dict.fromkeys(all_pdfs))  # Deduplicate
    
    for pdf_url in all_pdfs[:20]:  # Limit to first 20 PDFs to avoid timeout
        try:
            pr = http_get(pdf_url)
            pdf_text = extract_pdf_text(pr.content)
            if pdf_text and len(pdf_text) > 100:
                # Generate doc_id from PDF filename
                pdf_name = pdf_url.split("/")[-1].replace(".pdf", "").replace("%20", "_")
                docs_raw.append({
                    "doc_id": f"pdf_{pdf_name}",
                    "title": f"Analiz Raporu: {pdf_name}",
                    "url": pdf_url,
                    "text": pdf_text,
                    "pdf_links": [],
                    "images": [],
                })
                pdf_count += 1
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    # Chunk and write JSONL
    st.info("Parçalama (chunking) ve kayıt...")
    chunks: List[DocChunk] = []
    for d in docs_raw:
        full = d.get("text", "") or ""
        if not full.strip():
            continue
        chunks.extend(chunk_document(d["doc_id"], d.get("title", d["doc_id"]), d.get("url", ""), full))

    with open(DOCS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    # Build index
    st.info("Arama indeksi kuruluyor (TF-IDF)...")
    build_index_from_jsonl()

    meta = {
        "status": "ready",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": len(chunks),
        "pdf_count": pdf_count,
        "rock_count": len(rock_images),
        "source_main": KAYAC_MAIN,
    }
    json.dump(meta, open(META_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    return {"ok": True, "message": f"Tamamlandı. Toplam parça: {len(chunks)}, PDF: {pdf_count}"}


# -----------------------------
# Index / Retrieval
# -----------------------------
def load_chunks() -> List[DocChunk]:
    if not os.path.exists(DOCS_JSONL):
        return []
    out = []
    with open(DOCS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(DocChunk(**obj))
    return out


def build_index_from_jsonl():
    import pickle
    chunks = load_chunks()
    texts = [c.text for c in chunks]
    if not texts:
        return

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=120000
    )
    X = vectorizer.fit_transform(texts)

    with open(VECTORIZER_BIN, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MATRIX_BIN, "wb") as f:
        pickle.dump(X, f)


def load_index() -> Tuple[Optional[TfidfVectorizer], Any, List[DocChunk]]:
    import pickle
    chunks = load_chunks()
    if not chunks or not os.path.exists(VECTORIZER_BIN) or not os.path.exists(MATRIX_BIN):
        return None, None, chunks
    with open(VECTORIZER_BIN, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MATRIX_BIN, "rb") as f:
        X = pickle.load(f)
    return vectorizer, X, chunks


def retrieve(query: str, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
    vectorizer, X, chunks = load_index()
    if vectorizer is None or X is None or not chunks:
        return []

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        score = float(sims[i])
        if score <= 0:
            continue
        results.append((chunks[i], score))
    return results


# -----------------------------
# Enriched Rocks Database
# -----------------------------
def load_enriched_db() -> Dict[str, Any]:
    """Load enriched rocks database from JSON file."""
    if not os.path.exists(ENRICHED_ROCKS_JSON):
        return {}
    try:
        with open(ENRICHED_ROCKS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_enriched_db(data: Dict[str, Any]):
    """Save enriched rocks database to JSON file."""
    ensure_data_dir()
    with open(ENRICHED_ROCKS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_rock_enriched(doc_id: str) -> bool:
    """Check if a rock has already been enriched."""
    db = load_enriched_db()
    return doc_id in db and db[doc_id].get("ozellikler")


def get_rock_enrichment(doc_id: str) -> Optional[Dict[str, str]]:
    """Get enrichment data for a rock if exists."""
    db = load_enriched_db()
    return db.get(doc_id)


def load_rock_images() -> Dict[str, Any]:
    """Load rock images database from JSON file."""
    if not os.path.exists(ROCK_IMAGES_JSON):
        return {}
    try:
        with open(ROCK_IMAGES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_rocks_data() -> Dict[str, Any]:
    """Load pre-fetched rocks data from JSON file."""
    if not os.path.exists(ROCKS_DATA_JSON):
        return {}
    try:
        with open(ROCKS_DATA_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# -----------------------------
# Gemini AI Functions
# -----------------------------
def generate_answer_with_gemini(query: str, contexts: List[DocChunk]) -> str:
    """
    Generate answer using Google Gemini API.
    """
    if not GEMINI_AVAILABLE:
        return "Gemini kütüphanesi yüklü değil. 'pip install google-genai' komutunu çalıştırın."
    
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API anahtarı bulunamadı. .streamlit/secrets.toml dosyasını kontrol edin."
    
    ctx = "\n\n".join([f"[KAYNAK: {c.url}]\n{c.text}" for c in contexts[:6]])
    system = (
        "Sen Eskişehir Kayaç Parkı ziyaretçi rehberisin. "
        "Sadece verilen kaynak metinlere dayanarak yanıt ver. "
        "Kaynaklarda yoksa 'Bu bilgi paylaşılan kaynaklarda yer almıyor.' de. "
        "Yanıt dili Türkçedir. Kısa, açık, halkın anlayacağı şekilde yaz. "
        "Bilimsel terimleri kullanırken parantez içinde açıklamalar ekle."
    )
    
    user_message = f"Soru: {query}\n\nKaynaklar:\n{ctx}"
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini API hatası: {e}"


def enrich_rock_with_gemini(rock_name: str, rock_location: str, existing_text: str, pdf_text: str = "") -> Dict[str, str]:
    """
    Use Gemini to generate missing rock information:
    - Kayacın Özellikleri (properties)
    - Kayacın Kullanıldığı Alanlar (usage areas)
    """
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini kütüphanesi yüklü değil."}
    
    api_key = get_gemini_api_key()
    if not api_key:
        return {"error": "Gemini API anahtarı bulunamadı."}
    
    # Combine available context
    context_parts = []
    if existing_text:
        context_parts.append(f"Mevcut Bilgiler:\n{existing_text[:3000]}")
    if pdf_text:
        context_parts.append(f"PDF Analiz Raporu:\n{pdf_text[:3000]}")
    
    context = "\n\n".join(context_parts) if context_parts else "Ek bilgi mevcut değil."
    
    prompt = f"""Sen bir jeoloji ve kayaç bilimi uzmanısın. Aşağıdaki kayaç hakkında detaylı bilgi üret.

Kayaç Adı: {rock_name}
Çıkarıldığı Yer: {rock_location}

{context}

Lütfen aşağıdaki bilgileri Türkçe olarak detaylı şekilde üret:

1. **Kayacın Özellikleri**: Mineral bileşimi, dokusu (iri/ince taneli, camsı vb.), sertliği (Mohs skalası), rengi, oluşum koşulları (magmatik/tortul/metamorfik), jeolojik yaşı hakkında bilgi ver.

2. **Kayacın Kullanıldığı Alanlar**: İnşaat sektörü, dekoratif kullanım, endüstriyel uygulamalar, tarihsel/kültürel önem, Türkiye'deki kullanım örnekleri hakkında bilgi ver.

Yanıtını JSON formatında ver:
{{
    "ozellikler": "Kayacın özellikleri hakkında detaylı paragraf...",
    "kullanim_alanlari": "Kullanım alanları hakkında detaylı paragraf..."
}}"""
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        )
        
        result = json.loads(response.text)
        result["enriched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["rock_name"] = rock_name
        result["rock_location"] = rock_location
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract from plain text
        return {
            "ozellikler": response.text if 'response' in dir() else "Bilgi üretilemedi.",
            "kullanim_alanlari": "",
            "enriched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rock_name": rock_name,
            "rock_location": rock_location,
            "parse_error": True
        }
    except Exception as e:
        return {"error": str(e)}


def answer(query: str, top_k: int) -> Tuple[str, List[Tuple[DocChunk, float]]]:
    """Answer a query using Gemini AI with retrieved context."""
    hits = retrieve(query, top_k=top_k)
    contexts = [h[0] for h in hits]

    if not contexts:
        return "Bu bilgi paylaşılan kaynaklarda yer almıyor.", hits

    # Use Gemini for all answers
    try:
        answer_text = generate_answer_with_gemini(query, contexts)
        return answer_text, hits
    except Exception as e:
        # Fallback to extractive summary if KayAI fails
        bullets = []
        for c, score in hits[:3]:
            snippet = c.text
            if len(snippet) > 320:
                snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
            bullets.append(f"- {snippet}")
        msg = (
            f"⚠️ KayAI hatası: {e}\n\n"
            "Kaynaklarda bulunan en ilgili bölümler:\n"
            + "\n".join(bullets)
        )
        return msg, hits


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KayAI - Eskişehir Kayaç Parkı Rehberi", layout="wide")

st.title("🪨 KayAI - Eskişehir Kayaç Parkı Rehberi")
st.caption("AI Destekli Kayaç Bilgi Sistemi | 81 Kayaç")

tabs = st.tabs(["Kayaç Kataloğu", "Mini Sınav"])

# ---- Tab 1: Catalogue
with tabs[0]:
    st.subheader("Kayaç Kataloğu (001–081)")
    
    # Load pre-fetched rocks data
    rocks_data = load_rocks_data()
    rock_ids = sorted([k for k in rocks_data.keys() if k.startswith("rock_")])
    
    if not rock_ids:
        st.warning("⚠️ Katalog verisi bulunamadı. Lütfen 'fetch_data.py' scriptini çalıştırın.")
    else:
        selected = st.selectbox(
            "🪨 Kayaç Seçin", 
            rock_ids, 
            format_func=lambda x: f"{x.replace('rock_', '')} — {rocks_data[x]['title']}"
        )
        rock = rocks_data[selected]
        enrichment = get_rock_enrichment(selected)
        
        # Header with rock name
        st.markdown(f"### 🪨 {rock['title']}")
        st.markdown(f"🔗 [OGM Kaynak Sayfa]({rock['url']})")
        
        # Try to show rock image
        rock_images_db = load_rock_images()
        if selected in rock_images_db and rock_images_db[selected].get("images"):
            images = rock_images_db[selected]["images"]
            if images:
                try:
                    st.image(images[0], caption=f"{rock['title']} görseli", width=400)
                except Exception:
                    pass  # Image loading failed silently
        
        # Show enriched info if available
        if enrichment and enrichment.get("ozellikler"):
            # Two column layout for properties
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🔬 Kayacın Özellikleri")
                st.info(enrichment.get("ozellikler", "Bilgi yok."))
            
            with col2:
                st.markdown("#### 🏗️ Kullanım Alanları")
                st.info(enrichment.get("kullanim_alanlari", "Bilgi yok."))
        else:
            st.warning("⚠️ Bu kayaç henüz zenginleştirilmemiş. Lütfen 'enrich_all_rocks.py' scriptini çalıştırın.")
        
        # Rock Question Feature
        st.markdown("---")
        st.markdown("#### 💬 KayAI'ye Bu Kayaç Hakkında Soru Sor")
        
        # Initialize rock-specific chat
        rock_chat_key = f"rock_chat_{selected}"
        if rock_chat_key not in st.session_state:
            st.session_state[rock_chat_key] = []
        
        # Show previous Q&A for this rock
        for q, a in st.session_state[rock_chat_key]:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        
        rock_question = st.text_input(
            "Sorunuzu yazın:", 
            placeholder=f"{rock['title']} hakkında ne öğrenmek istersiniz?",
            key=f"rock_q_{selected}"
        )
        
        if st.button("🤖 KayAI'ye Sor", key=f"ask_{selected}"):
            if rock_question:
                with st.spinner("KayAI yanıt üretiyor..."):
                    # Build context from rock data and enrichment
                    context = f"Kayaç: {rock['title']}\n"
                    context += f"Kaynak: {rock.get('text', '')}\n"
                    if enrichment:
                        context += f"Özellikleri: {enrichment.get('ozellikler', '')}\n"
                        context += f"Kullanım Alanları: {enrichment.get('kullanim_alanlari', '')}\n"
                    
                    # Create a DocChunk-like object for the answer function
                    from dataclasses import dataclass
                    @dataclass
                    class TempChunk:
                        doc_id: str
                        title: str
                        url: str
                        chunk_id: str
                        text: str
                    
                    temp_chunk = TempChunk(
                        doc_id=selected,
                        title=rock['title'],
                        url=rock['url'],
                        chunk_id=f"{selected}::enriched",
                        text=context
                    )
                    
                    # Generate answer
                    rock_answer = generate_answer_with_gemini(
                        f"{rock['title']} hakkında: {rock_question}",
                        [temp_chunk]
                    )
                    
                    # Save to session state
                    st.session_state[rock_chat_key].append((rock_question, rock_answer))
                    st.rerun()
            else:
                st.warning("Lütfen bir soru yazın.")

# ---- Tab 2: Mini Sınav
with tabs[1]:
    st.subheader("🎮 Mini Sınav - Kayaç Bilgisi Testi")
    st.write("81 kayaç hakkında bilginizi test edin! Her sınav 5 görsel sorudan oluşur.")
    
    import random
    
    # Load rocks data for sınav
    rocks_data_quiz = load_rocks_data()
    enriched_db_quiz = load_enriched_db()
    rock_images_quiz = load_rock_images()
    
    rock_list = [k for k in rocks_data_quiz.keys() if k.startswith("rock_")]
    
    # Filter rocks that have images
    rocks_with_images = [r for r in rock_list if r in rock_images_quiz and rock_images_quiz[r].get("images")]
    
    if not rocks_with_images:
        st.warning("⚠️ Sınav için görsel bulunamadı.")
    else:
        # Initialize sınav state
        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
        
        def generate_quiz():
            """Generate 5 image-based multiple choice questions about rocks."""
            questions = []
            # Pick 5 random rocks that have images
            selected_rocks = random.sample(rocks_with_images, min(5, len(rocks_with_images)))
            
            for rock_id in selected_rocks:
                rock = rocks_data_quiz[rock_id]
                rock_title = rock.get("title", "Bilinmeyen Kayaç")
                rock_images = rock_images_quiz.get(rock_id, {}).get("images", [])
                
                if not rock_images:
                    continue
                
                # Get other rocks for wrong answers
                other_rocks = [r for r in rock_list if r != rock_id]
                wrong_choices = random.sample(other_rocks, min(3, len(other_rocks)))
                wrong_titles = [rocks_data_quiz[r].get("title", "Bilinmeyen") for r in wrong_choices]
                
                question = {
                    "question": "Görseldeki kayaç hangisidir?",
                    "image_url": rock_images[0],
                    "correct": rock_title,
                    "options": [rock_title] + wrong_titles,
                    "rock_id": rock_id
                }
                
                # Shuffle options
                random.shuffle(question["options"])
                questions.append(question)
            
            return questions
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎲 Yeni Sınav Oluştur", type="primary"):
                st.session_state.quiz_questions = generate_quiz()
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.rerun()
        
        with col2:
            if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
                if st.button("✅ Cevapları Kontrol Et"):
                    st.session_state.quiz_submitted = True
                    # Calculate score
                    score = 0
                    for i, q in enumerate(st.session_state.quiz_questions):
                        user_ans = st.session_state.quiz_answers.get(i, "")
                        if user_ans == q["correct"]:
                            score += 1
                    st.session_state.quiz_score = score
                    st.rerun()
        
        # Display questions
        if st.session_state.quiz_questions:
            st.markdown("---")
            
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"### Soru {i+1}")
                
                # Show rock image with fixed width
                if q.get("image_url"):
                    try:
                        st.image(q["image_url"], width=300)
                    except Exception:
                        st.warning("Görsel yüklenemedi")
                
                st.markdown(f"**{q['question']}**")
                
                # Radio buttons for options
                if st.session_state.quiz_submitted:
                    # Show results
                    user_ans = st.session_state.quiz_answers.get(i, "")
                    correct_ans = q["correct"]
                    
                    for opt in q["options"]:
                        if opt == correct_ans:
                            st.success(f"✅ {opt}" + (" (Sizin cevabınız)" if opt == user_ans else ""))
                        elif opt == user_ans:
                            st.error(f"❌ {opt} (Sizin cevabınız)")
                        else:
                            st.write(f"○ {opt}")
                else:
                    # Show radio buttons
                    selected = st.radio(
                        f"Cevabınız:",
                        q["options"],
                        key=f"quiz_q_{i}",
                        index=None,
                        label_visibility="collapsed"
                    )
                    if selected:
                        st.session_state.quiz_answers[i] = selected
                
                st.markdown("---")
            
            # Show final score
            if st.session_state.quiz_submitted:
                score = st.session_state.quiz_score
                total = len(st.session_state.quiz_questions)
                
                if score == total:
                    st.balloons()
                    st.success(f"🏆 Tebrikler! Tüm soruları doğru cevapladınız! Skor: {score}/{total}")
                elif score >= total * 0.6:
                    st.info(f"👍 İyi iş! Skor: {score}/{total}")
                else:
                    st.warning(f"📚 Biraz daha çalışmanız gerekiyor. Skor: {score}/{total}")
        else:
            st.info("👆 Sınav başlatmak için 'Yeni Sınav Oluştur' butonuna tıklayın.")


# Footer
st.markdown("---")
st.caption("🪨 Eskişehir Kayaç Parkı - Yapıy Zekâ Destekli Bilgi Sistemi")
