# KayAI - Eskisehir Kayac Parki AI Rehberi

KayAI, Eskisehir Kayac Parki verilerini cekip zenginlestiren ve Streamlit arayuzu ile soru-cevap yapabilen bir Python uygulamasidir.

## Ozellikler
- OGM kaynaklarindan kayac verisi cekme (`fetch_data.py`)
- Gemini ile kayac bilgisini zenginlestirme (`enrich_all_rocks.py`)
- Streamlit ile arama ve soru-cevap arayuzu (`app.py`)

## Kurulum
1. Python 3.10+ kurulu olmali.
2. Sanal ortam olusturun:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Bagimliliklari yukleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Konfigurasyon
Gemini anahtari kod icine yazilmaz. Anahtari ortam degiskeni veya Streamlit secrets ile verin.

### Secenek 1: Environment Variable (onerilen)
```bash
setx GEMINI_API_KEY "BURAYA_KENDI_ANAHTARIN"
```
Yeni terminal acip dogrulayin:
```bash
echo %GEMINI_API_KEY%
```

### Secenek 2: Streamlit Secrets
` .streamlit/secrets.toml.example ` dosyasini referans alip `.streamlit/secrets.toml` olusturun.

## Calistirma
### 1) Veriyi cek
```bash
python fetch_data.py
```

### 2) Gemini ile zenginlestir
```bash
python enrich_all_rocks.py
```

### 3) Uygulamayi ac
```bash
streamlit run app.py
```

## Guvenlik Notlari
- API anahtarlarini asla repoya commit etmeyin.
- `.gitignore`, `.env`, `.streamlit/secrets.toml` ve olasi sertifika dosyalarini disarida tutacak sekilde ayarlanmistir.
- Daha once paylasilmis bir Gemini anahtari varsa hemen iptal edin (revoke/rotate) ve yeni anahtar olusturun.
- Public repoda gercek veri veya hassas dosya paylasmadan once `git status` ile kontrol edin.

## Proje Yapisi
- `app.py`: Streamlit ana uygulamasi
- `fetch_data.py`: Veri cekme scripti
- `enrich_all_rocks.py`: Gemini ile zenginlestirme scripti
- `requirements.txt`: Python bagimliliklari
- `data_cache/`: Uretilen/veri cache klasoru (repo disinda tutulur)
