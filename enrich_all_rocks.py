"""
Eskişehir Kayaç Parkı - Tüm Kayaçları Gemini ile Zenginleştirme Scripti
Bu script tüm 81 kayacı Gemini AI ile zenginleştirir ve enriched_rocks.json'a kaydeder.
"""

import os
import json
import time
from google import genai
from google.genai import types

# Configuration
DATA_DIR = "data_cache"
ROCKS_DATA_JSON = os.path.join(DATA_DIR, "rocks_data.json")
ENRICHED_ROCKS_JSON = os.path.join(DATA_DIR, "enriched_rocks.json")

GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_api_key():
    """Read API key from environment variables only."""
    return os.environ.get("GEMINI_API_KEY", "").strip()


def load_rocks_data():
    """Load pre-fetched rocks data from JSON file."""
    if not os.path.exists(ROCKS_DATA_JSON):
        return {}
    with open(ROCKS_DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_enriched_db():
    """Load existing enriched database."""
    if not os.path.exists(ENRICHED_ROCKS_JSON):
        return {}
    try:
        with open(ENRICHED_ROCKS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_enriched_db(data):
    """Save enriched database to JSON file."""
    with open(ENRICHED_ROCKS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def enrich_rock_with_gemini(rock_name, rock_location, existing_text):
    """Use Gemini to generate rock properties and usage areas."""
    
    prompt = f"""Sen bir jeoloji ve kayaç bilimi uzmanısın. Aşağıdaki kayaç hakkında detaylı bilgi üret.

Kayaç Adı: {rock_name}
Çıkarıldığı Yer: {rock_location}

Mevcut Bilgiler:
{existing_text[:2000] if existing_text else 'Ek bilgi mevcut değil.'}

Lütfen aşağıdaki bilgileri Türkçe olarak detaylı şekilde üret:

1. **Kayacın Özellikleri**: Mineral bileşimi, dokusu (iri/ince taneli, camsı vb.), sertliği (Mohs skalası), rengi, oluşum koşulları (magmatik/tortul/metamorfik), jeolojik yaşı hakkında bilgi ver.

2. **Kayacın Kullanıldığı Alanlar**: İnşaat sektörü, dekoratif kullanım, endüstriyel uygulamalar, tarihsel/kültürel önem, Türkiye'deki kullanım örnekleri hakkında bilgi ver.

Yanıtını JSON formatında ver:
{{
    "ozellikler": "Kayacın özellikleri hakkında detaylı paragraf...",
    "kullanim_alanlari": "Kullanım alanları hakkında detaylı paragraf..."
}}"""
    
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            return {
                "error": "GEMINI_API_KEY bulunamadi. Anahtari environment variable olarak tanimlayin."
            }

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
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


def main():
    print("=" * 60)
    print("Eskişehir Kayaç Parkı - Gemini Zenginleştirme")
    print("=" * 60)
    print()
    
    # Load rocks data
    rocks_data = load_rocks_data()
    if not rocks_data:
        print("HATA: rocks_data.json bulunamadı. Önce fetch_data.py çalıştırın.")
        return
    
    # Load existing enriched data
    enriched_db = load_enriched_db()
    
    rock_ids = sorted([k for k in rocks_data.keys() if k.startswith("rock_")])
    total = len(rock_ids)
    
    print(f"Toplam {total} kayaç bulundu.")
    print(f"Zaten zenginleştirilmiş: {len([k for k in enriched_db if enriched_db.get(k, {}).get('ozellikler')])}")
    print("-" * 40)
    print()
    
    enriched_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, rock_id in enumerate(rock_ids, 1):
        rock = rocks_data[rock_id]
        rock_title = rock.get("title", "Bilinmeyen Kayaç")
        
        # Skip if already enriched
        if rock_id in enriched_db and enriched_db[rock_id].get("ozellikler"):
            print(f"  [{i:03d}/{total}] ⏭️ {rock_title} (zaten zenginleştirilmiş)")
            skipped_count += 1
            continue
        
        print(f"  [{i:03d}/{total}] 🤖 {rock_title} zenginleştiriliyor...", end=" ", flush=True)
        
        # Extract location info
        rock_text = rock.get("text", "")
        rock_location = "Belirtilmemiş"
        if "Çıkarıldığı Yer" in rock_text:
            rock_location = rock_text[:300]
        
        # Call Gemini
        result = enrich_rock_with_gemini(
            rock_name=rock_title,
            rock_location=rock_location,
            existing_text=rock_text
        )
        
        if "error" in result:
            print(f"❌ Hata: {result['error']}")
            error_count += 1
        else:
            enriched_db[rock_id] = result
            save_enriched_db(enriched_db)  # Save after each success
            print("✅")
            enriched_count += 1
        
        # Rate limiting - wait between requests
        time.sleep(1.5)
    
    print()
    print("=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print(f"  • Yeni zenginleştirilen: {enriched_count}")
    print(f"  • Atlanan (zaten hazır): {skipped_count}")
    print(f"  • Hata: {error_count}")
    print(f"  • Toplam zenginleştirilmiş: {len([k for k in enriched_db if enriched_db.get(k, {}).get('ozellikler')])}/{total}")
    print()
    print(f"Dosya kaydedildi: {ENRICHED_ROCKS_JSON}")


if __name__ == "__main__":
    main()
