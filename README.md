# KayAI - Eskişehir Kayaç Parkı Dijital Rehberi

Bu çalışma, Eskişehir Orman Bölge Müdürlüğü bünyesinde yer alan ve Türkiye'nin farklı bölgelerinden derlenen 81 farklı kayaç örneğini barındıran Kayaç Parkı'nın, ziyaretçiler tarafından daha kolay anlaşılabilir ve etkileşimli biçimde deneyimlenmesini amaçlayan dijital bir rehber yaklaşımını ele almaktadır.

Kayaç Parkı, jeoloji ile ormancılık arasındaki ilişkiyi somut örneklerle sunan önemli bir açık alan öğrenme ortamı olmasına rağmen; kayaç çeşitliliğinin fazlalığı ve mevcut bilgilendirme tabelalarının sınırlı içeriği, özellikle genel ziyaretçi kitlesi için bilginin etkin aktarımını zorlaştırabilmektedir.

Bu çalışmanın temel amacı, parkta sergilenen kayaçlara ilişkin bilgilerin ziyaretçilerin park içindeki deneyimiyle doğrudan ilişkilendirildiği, sade, anlaşılır ve etkileşimli bir dijital yapı aracılığıyla sunulmasıdır.

Bu kapsamda geliştirilen yaklaşımda ziyaretçi, park gezisi sırasında mobil cihazı üzerinden KayAI adlı dijital rehberi kullanarak kayaçlarla birebir etkileşime geçebilmektedir. Ziyaretçi, karşısında bulunan kayacı katalogdan seçtiğinde ilgili kayaç görseli, temel fiziksel özellikleri ve kullanım alanları ekranda görüntülenmekte; doğal dilde yönelttiği sorular aracılığıyla kayaçların oluşum süreçleri ve ormancılıkla ilişkisi hakkında anında açıklamalar alabilmektedir. Böylece ziyaretçi, gezip gördüğü kayaçlar üzerinden aktif bir öğrenme sürecine dahil olmaktadır.

## Yöntem

Çalışmada yöntem olarak, Orman Genel Müdürlüğü tarafından kamuya açık biçimde sunulan Kayaç Parkı içerikleri temel alınmıştır. Parkta yer alan 81 kayaç örneğine ait bilgiler; köken, fiziksel özellikler, ayrışma süreçleri ve orman-toprak ilişkisi bağlamında sadeleştirilmiş ve yapılandırılmıştır.

Bu içerikler, ziyaretçilerin hem belirli bir kayacı seçerek bilgi edinmesine hem de genel kavramsal sorular yöneltmesine imkan tanıyan etkileşimli bir dijital rehber aracılığıyla sunulmuştur.

## Sistem Bileşenleri

Geliştirilen sistem iki ana bileşenden oluşmaktadır:

- Kayaç Kataloğu: Ziyaretçilerin parkta yer alan kayaçları karşılaştırmalı olarak incelemesine olanak sağlar.
- Mini Sınav: Görsel destekli kısa sorular aracılığıyla öğrenilen bilgilerin pekiştirilmesini hedefler.

Yapay zeka teknolojileri bu çalışmada temel amaç olarak değil, bilginin aktarımını kolaylaştıran bir destek aracı olarak ele alınmıştır. Soru-cevap mekanizması, jeolojik ve ormancılık terimlerini sadeleştirerek halkın anlayabileceği bir anlatım sunmakta ve ziyaretçilerin park gezisini etkileşimli bir öğrenme deneyimine dönüştürmektedir.

## Sonuç

Bu çalışma, Kayaç Parkı'nın kamuoyu bilinçlendirme ve eğitim misyonunu güçlendiren, ziyaretçinin uygulamayı zihninde kolayca canlandırabildiği ve deneyimleyebildiği bir dijital rehber modeli ortaya koymakta; benzer tematik açık alan uygulamaları için de uygulanabilir bir örnek sunmaktadır.

## Teknik Kurulum

1. Python 3.10+ kurulu olmalıdır.
2. Sanal ortam oluşturun:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## Konfigürasyon

Gemini anahtarı kod içine yazılmaz. Anahtarı yalnızca ortam değişkeni üzerinden verin:

```bash
setx GEMINI_API_KEY "BURAYA_KENDI_ANAHTARIN"
```

Yeni terminal açıp doğrulayın:

```bash
echo %GEMINI_API_KEY%
```

## Çalıştırma

1. Veriyi çekin:

```bash
python fetch_data.py
```

2. Kayaç bilgilerini zenginleştirin:

```bash
python enrich_all_rocks.py
```

3. Uygulamayı başlatın:

```bash
streamlit run app.py
```

## Güvenlik Notları

- API anahtarlarını repoya commit etmeyin.
- .gitignore içinde .streamlit ve .env dosyaları hariç tutulmuştur.
- Daha önce paylaşılmış anahtar varsa derhal iptal edip yenisini üretin.
