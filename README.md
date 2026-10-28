# 🌌 Samanyolu Senfonisi (Milky Way Symphony)

Uzay verilerini **sonifikasyon (data sonification)** yöntemiyle dinlenebilir bir **ambient / space müzik** altyapısına (MIDI) dönüştüren, modüler ve nesne yönelimli bir Python projesi.

Işık eğrileri, pulsar radyo sinyalleri ve kütleçekim dalgaları; müzik teorisi kurallarına bağlı bir algoritmayla notalara, ritimlere ve enstrüman rollerine eşlenir. Amaç kaotik bir gürültü değil, estetik ve sinematik bir parça üretmektir.

> Not: Proje şu an **yapay (mock) veriyle** çalışır. Gerçek TESS/Kepler, ATNF ve LIGO verilerine bağlanmak için hazırdır (bkz. [Gerçek veriye geçiş](#gerçek-veriye-geçiş)).

## ✨ Özellikler

- 3 farklı astronomik olgunun NumPy ile simülasyonu
- Tüm notalar **C Minor Pentatonic** (varsayılan) veya **C Natural Minor (Aeolian)** skalasına zorlanır (*snap to scale*)
- 60 veya 70 BPM ile derin uzay hissi
- 3 ayrı MIDI track'i, tek bir `.mid` dosyasında birleştirilir
- Parçanın başında ve sonunda yumuşak giriş/çıkış (fade)
- Ham veri ile nota/frekans eşleşmesini gösteren 3 panelli `matplotlib` grafiği
- Her adımı açıklayan log mesajları (Kaggle'da süreç adım adım izlenebilir)
- OOP yapı, PEP 8 uyumlu, yorum satırlarıyla açıklanmış kod

## 🎼 Veri → Müzik Eşlemesi

| Track | Rol | Veri kaynağı | Eşleme | Enstrüman (General MIDI) |
|-------|-----|--------------|--------|--------------------------|
| 1 | Yıldız melodisi | Işık eğrisi (TESS/Kepler) | Min-Max normalizasyon → C4–C6 skala notaları. Parlaklık artınca perde yükselir | Akustik Piyano (program 0) |
| 2 | Pulsar ritmi | Pulsar radyo sinyali (ATNF) | Tepe noktaları (peak) algılanır, 16'lık ızgaraya oturtulur. Güçlü pulslar C2 kick, diğerleri hi-hat | Perküsyon (kanal 10) |
| 3 | Kara delik atmosferi | Kütleçekim dalgası (LIGO) | C1–C3 aralığına eşlenir, ölçü başına uzun nota + skalaya uyan bir beşli | Warm Pad (program 89) |

### Müzikal kısıtlar

- **Skala:** `minor_pentatonic` → C, Eb, F, G, Bb  ·  `natural_minor` → C, D, Eb, F, G, Ab, Bb
- **Tempo:** 60 veya 70 BPM (farklı bir değer `ValueError` verir)
- **Nota havuzu:** Veri ne olursa olsun nota, havuzdaki en yakın skala notasına yuvarlanır; böylece kakofoni oluşmaz.

## 🛰️ Mock Veri Setleri

| Veri | Simüle edilen olgu |
|------|--------------------|
| `light_curve` | Yavaş parlaklık salınımı, periyodik gezegen geçişi (transit) düşüşleri, ani parlamalar (flare) ve mikro dalgalanmalar |
| `pulsar` | Düzenli, keskin radyo vuruşları. Her 4. puls vurgulu, nadiren "nulling" (puls atlaması) |
| `gravity` | Düşük frekanslı salınımlar, birleşme öncesi yükselen chirp, ani çöküş ve sönümlü ringdown |

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.8+
- `numpy`, `matplotlib`, `mido`

```bash
pip install numpy matplotlib mido
```

> Kod, `mido` yüklü değilse onu otomatik kurmayı da dener.

### Yerelde

```bash
python milky_way_symphony.py
```

### Kaggle Notebook'ta

1. Yeni bir notebook açıp dosyanın içeriğini bir hücreye yapıştırın.
2. Gerekirse hücrenin en başına şunu ekleyin:

   ```python
   !pip install mido
   ```

3. Hücreyi çalıştırın. Çıktılar `/kaggle/working` klasörüne yazılır.

## 📦 Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `milky_way_symphony.mid` | 3 track'li MIDI dosyası |
| `milky_way_symphony.png` | 3 panelli görsel |

Grafik panelleri:

1. **Ham uzay verisi:** üç serinin normalize edilmiş hali
2. **Pitch vs. Time:** melodi, bas/pad ve pulsar vuruşlarını gösteren piyano-roll
3. **Frekans vs. Time:** notaların Hz karşılığı (logaritmik eksen)

MIDI dosyasını DAW'a (Ableton, FL Studio, GarageBand, LMMS...) veya bir MIDI çalıcıya yükleyip istediğiniz sentezleyiciyle (reverb'li piyano, geniş pad vb.) dinleyebilirsiniz.

## ⚙️ Yapılandırma

Tüm ayarlar `SymphonyConfig` içinde toplanmıştır:

```python
from milky_way_symphony import MilkyWaySymphony, SymphonyConfig

config = SymphonyConfig(
    bpm=70,                    # 60 veya 70
    bars=48,                   # ölçü sayısı (parça süresini belirler)
    scale="natural_minor",     # "minor_pentatonic" | "natural_minor"
    seed=7,                    # farklı mock veri için
)
MilkyWaySymphony(config).run()
```

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `bpm` | `60` | Tempo (60 veya 70) |
| `bars` | `32` | Ölçü sayısı (60 BPM'de ≈ 2 dk 8 sn) |
| `scale` | `"minor_pentatonic"` | Kullanılacak skala |
| `n_samples` | `2048` | Her veri serisinin örnek sayısı |
| `seed` | `42` | Rastgelelik tohumu (tekrarlanabilirlik) |
| `melody_range` | `(60, 84)` | Melodi aralığı (C4–C6) |
| `bass_range` | `(24, 48)` | Bas/pad aralığı (C1–C3) |

## 🧱 Proje Yapısı

| Sınıf | Görevi |
|-------|--------|
| `SymphonyConfig` | Tüm ayarlar |
| `SignalTools` | Normalizasyon, yeniden örnekleme, yumuşatma, tepe noktası tespiti |
| `MockDataGenerator` | Yapay uzay verisi üretimi |
| `ScaleMapper` | Değerleri skala nota havuzuna eşleme |
| `MelodySonifier` | Track 1 (yıldız melodisi) |
| `PulsarSonifier` | Track 2 (pulsar ritmi) |
| `BassPadSonifier` | Track 3 (bas / pad) |
| `MidiComposer` | Track'leri birleştirip MIDI olarak dışa aktarma |
| `SymphonyVisualizer` | 3 panelli grafik |
| `MilkyWaySymphony` | Tüm akışı yöneten ana sınıf |

## 🔭 Gerçek veriye geçiş

`MilkyWaySymphony.run()` hazır bir veri sözlüğü kabul eder. Her seri 1 boyutlu bir NumPy dizisi olmalıdır:

```python
data = {
    "light_curve": tess_flux,        # TESS / Kepler ışık eğrisi
    "pulsar": pulsar_intensity,      # ATNF tabanlı pulsar sinyali
    "gravity": ligo_strain,          # LIGO kütleçekim dalgası
}
MilkyWaySymphony().run(data=data)
```

Gerçek veriler farklı uzunlukta olabilir; kod her seriyi parça süresine göre otomatik yeniden örnekler. Pulsar verisinde tepe noktaları eşik değeriyle (ortalama + 0.5 × standart sapma) bulunur; gerçek veride bu eşiği ayarlamanız gerekebilir.

## 🗺️ Yol Haritası

- [ ] Gerçek API entegrasyonları (TESS/Kepler, ATNF, LIGO)
- [ ] Işık eğrisi için arpej katmanı
- [ ] Kullanıcı seçimli akor ilerlemeleri
- [ ] WAV/MP3 dışa aktarımı (FluidSynth ile)

## 📄 Lisans

Bu proje açık kaynaktır. Yayınlamadan önce deponuza bir `LICENSE` dosyası ekleyin (ör. MIT).
