# MLP ile Sınıflandırma: EDA, Ön İşleme, Hiperparametre Optimizasyonu ve Nested Cross-Validation

Yapay Sinir Ağları dersi — Yüksek Lisans Ara Sınav Projesi
Hakan Kızılhan

Bu repository, UCI Bank Marketing (`bank-additional.csv`) veri setinde vadeli mevduat aboneliğini (y = yes/no) tahmin eden MLP (Multi-Layer Perceptron, Çok Katmanlı Algılayıcı) modellerinin; ön işleme, mimari, optimizer ve eğitim hiperparametrelerine duyarlılığını veri sızıntısı (data leakage) olmadan, nested CV (Nested Cross-Validation, iç içe çapraz doğrulama) ile inceleyen deneysel çalışmayı içerir.

## Veri seti

- Kaynak: Moro, S., Rita, P. ve Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Lisans: CC BY 4.0 (Creative Commons Attribution 4.0 International). Dosya, kaynağa atıfla `data/bank-additional.csv` altında paylaşılmıştır.
- Yalnızca `bank-additional.csv` kullanılır (4.119 gözlem, 20 girdi özelliği). `duration` özelliği, görüşme bitmeden bilinmediği için tüm modellerden çıkarılmıştır.

## Klasör yapısı

| Yol | İçerik |
|---|---|
| `01_eda.ipynb` | Development/test ayrımı ve yalnızca development verisinde EDA (Exploratory Data Analysis, keşifsel veri analizi) |
| `02_nested_cv.ipynb` | Fold-local ön işleme, manuel (M1–M6) ve otomatik (R1–R6) arama, nested CV |
| `03_final_test.ipynb` | Nihai seçim, seed kararlılığı, kilitli test tahminleri, test metrikleri, bootstrap |
| `configs/` | Seed'ler, eğitim ayarları, arama uzayı, manuel ve otomatik adaylar |
| `results/` | Bölünme, deney, kat sonuçları, tahminler ve nihai metrikler (CSV/JSON) |
| `figures/` | Raporda kullanılan, kodla üretilmiş grafikler |
| `report.pdf` | Bilimsel teknik rapor |

## Kurulum

```bash
git clone https://github.com/hkizilhan/YSA-vize-odev.git
cd YSA-vize-odev
pip install -r requirements.txt
```

## Çalıştırma sırası

Notebook'lar sırayla, baştan sona çalıştırılmalıdır:

1. `01_eda.ipynb`
2. `02_nested_cv.ipynb`
3. `03_final_test.ipynb`

## Seed politikası

Temel seed = öğrenci numarasının son beş hanesi + 2026 = 29039 + 2026 = **31065**. Alt seed'ler `alt_seed = temel_seed + offset` kuralıyla türetilir; tam liste `configs/seeds.json` dosyasındadır.

## Sistem ve kitaplık sürümleri

- CPU: AMD Ryzen 5 5500U (12 iş parçacığı), GPU kullanılmadı
- İşletim sistemi: Fedora Linux 45
- Python 3.13.9; paket sürümleri `requirements.txt` dosyasındadır.

## Sonuçlar

_Deneyler tamamlandığında güncellenecektir._
