#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samanyolu Senfonisi (Milky Way Symphony)
========================================

Uzay verilerini (ışık eğrisi, pulsar radyo sinyali, kütleçekim dalgası)
sonifikasyon yöntemiyle dinlenebilir bir ambient / space müzik altyapısına
(MIDI) dönüştüren, modüler ve nesne yönelimli bir Python projesi.

Veri -> Müzik eşlemesi
----------------------
* TRACK 1  Yıldız Melodisi : Işık eğrisi   -> C4-C6 skala notaları (piyano)
* TRACK 2  Pulsar Ritmi    : Radyo pulsları -> C2 kick / hi-hat vuruşları
* TRACK 3  Kara Delik Pad  : Kütleçekim dalgası -> C1-C3 uzun bas / pad notaları

Müzikal kısıtlar
----------------
* Skala : C Minor Pentatonic (varsayılan) veya C Natural Minor (Aeolian)
* Tempo : 60 veya 70 BPM
* Tüm perdeler skalaya "snap" edilir; kakofoni oluşmaz.

Kaggle Notebook notu
--------------------
Eksik olabilecek 'mido' paketi için hücrenin en başına şu satırı ekleyin
(bir .py dosyasında geçersiz olduğu için yorum satırı olarak bırakıldı):

!pip install mido

Aşağıdaki ``ensure_dependencies`` fonksiyonu, paket yüklü değilse kurulumu
otomatik olarak da dener; yani dosya hem Kaggle'da hem yerelde çalışır.

Gerçek veriye geçiş
-------------------
``MockDataGenerator`` yerine TESS/Kepler (ışık eğrisi), ATNF (pulsar) veya
LIGO (kütleçekim dalgası) verilerini okuyan bir sınıf yazıp, aynı biçimde
(1 boyutlu NumPy dizisi) ``MilkyWaySymphony.run`` içine verebilirsiniz.
"""

# !pip install mido

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def ensure_dependencies(packages: Sequence[str] = ("mido",)) -> None:
    """Eksik paketleri (ör. Kaggle'da mido) otomatik olarak kurar."""
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"[KURULUM] '{package}' bulunamadı, pip ile kuruluyor...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )


ensure_dependencies()

import matplotlib.pyplot as plt  # noqa: E402
import mido  # noqa: E402


# --------------------------------------------------------------------------
# Sabitler ve yardımcılar
# --------------------------------------------------------------------------
# Skala aralıkları (C köküne göre yarım ses cinsinden)
SCALES: Dict[str, Tuple[int, ...]] = {
    "minor_pentatonic": (0, 3, 5, 7, 10),        # C Eb F G Bb
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),     # C D Eb F G Ab Bb (Aeolian)
}

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# General MIDI davul haritası (C2 = MIDI 36, orta C = C4 = 60 kuralı)
KICK_NOTE = 36        # C2
HIHAT_NOTE = 42       # F#2 / Gb2 (closed hi-hat)

# MIDI kanalları (0 tabanlı). Kanal 9 = General MIDI perküsyon kanalı.
CH_MELODY, CH_BASS, CH_DRUMS = 0, 1, 9


def log(step: str, message: str) -> None:
    """Kaggle kullanıcısının süreci adım adım görmesi için loglama."""
    print(f"[{step}] {message}")


def midi_to_name(note: int) -> str:
    """MIDI numarasını nota adına çevirir (60 -> 'C4')."""
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def midi_to_freq(note: float) -> float:
    """MIDI numarasını Hz cinsinden frekansa çevirir (A4 = 440 Hz)."""
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def default_output_dir() -> Path:
    """Kaggle'da /kaggle/working, aksi halde çalışma dizini."""
    kaggle_dir = Path("/kaggle/working")
    return kaggle_dir if kaggle_dir.exists() else Path.cwd()


@dataclass(frozen=True)
class SymphonyConfig:
    """Projenin tüm ayarları tek yerde tutulur."""

    bpm: int = 60                       # 60 veya 70
    ticks_per_beat: int = 480
    bars: int = 32                      # 32 ölçü @60 BPM ~ 2 dakika 8 saniye
    beats_per_bar: int = 4
    n_samples: int = 2048               # her veri serisinin örnek sayısı
    scale: str = "minor_pentatonic"     # "minor_pentatonic" | "natural_minor"
    seed: int = 42
    melody_range: Tuple[int, int] = (60, 84)   # C4 - C6
    bass_range: Tuple[int, int] = (24, 48)     # C1 - C3
    output_dir: Path = field(default_factory=default_output_dir)
    midi_name: str = "milky_way_symphony.mid"
    figure_name: str = "milky_way_symphony.png"

    def __post_init__(self) -> None:
        if self.bpm not in (60, 70):
            raise ValueError("BPM değeri 60 veya 70 olmalıdır.")
        if self.scale not in SCALES:
            raise ValueError(f"Bilinmeyen skala: {self.scale}")

    @property
    def total_beats(self) -> int:
        return self.bars * self.beats_per_bar

    @property
    def duration_seconds(self) -> float:
        return self.total_beats * 60.0 / self.bpm

    def beat_to_seconds(self, beat: float) -> float:
        return beat * 60.0 / self.bpm


@dataclass
class NoteEvent:
    """Zaman çizgisi üzerindeki tek bir müzikal olay (vuruş cinsinden)."""

    start_beat: float
    duration_beats: float
    pitch: int
    velocity: int
    channel: int


# --------------------------------------------------------------------------
# Sinyal işleme araçları
# --------------------------------------------------------------------------
class SignalTools:
    """Normalizasyon, yeniden örnekleme, yumuşatma ve tepe noktası tespiti."""

    @staticmethod
    def min_max_normalize(x: np.ndarray) -> np.ndarray:
        """Diziyi [0, 1] aralığına ölçekler (Min-Max Normalization)."""
        x = np.asarray(x, dtype=float)
        span = x.max() - x.min()
        if span == 0:
            return np.zeros_like(x)
        return (x - x.min()) / span

    @staticmethod
    def moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Kenar değerlerini koruyarak hareketli ortalama uygular."""
        if window <= 1:
            return np.asarray(x, dtype=float).copy()
        left = window // 2
        right = window - 1 - left
        padded = np.pad(x, (left, right), mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")

    @staticmethod
    def resample_mean(x: np.ndarray, n_bins: int) -> np.ndarray:
        """Diziyi n_bins parçaya bölüp her parçanın ortalamasını alır."""
        edges = np.linspace(0, len(x), n_bins + 1).astype(int)
        return np.array(
            [x[a:max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:])]
        )

    @staticmethod
    def find_peaks(x: np.ndarray, threshold: float) -> np.ndarray:
        """Eşik üstündeki yerel maksimumların indekslerini döndürür."""
        padded = np.pad(x, 1, constant_values=-np.inf)
        left, right = padded[:-2], padded[2:]
        mask = (x > left) & (x >= right) & (x >= threshold)
        return np.where(mask)[0]


def fade_gain(beat: float, total_beats: float,
              fade_beats: float = 8.0, floor: float = 0.35) -> float:
    """Parçanın başında ve sonunda yumuşak giriş/çıkış (fade) katsayısı."""
    g = min(beat / fade_beats, (total_beats - beat) / fade_beats, 1.0)
    return floor + (1.0 - floor) * max(g, 0.0)


def to_velocity(value: float, beat: float, cfg: SymphonyConfig) -> int:
    """Ham velocity değerini fade katsayısıyla çarpıp 1-127'ye sıkıştırır."""
    return int(np.clip(value * fade_gain(beat, cfg.total_beats), 1, 127))


# --------------------------------------------------------------------------
# 1. Veri modelleme ve mock data (giriş katmanı)
# --------------------------------------------------------------------------
class MockDataGenerator:
    """Gerçek API'lara bağlanmadan önce test için yapay uzay verisi üretir."""

    def __init__(self, cfg: SymphonyConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def stellar_light_curve(self) -> np.ndarray:
        """TESS/Kepler benzeri yıldız ışık eğrisi (0-1 arası).

        Yavaş parlaklık salınımı + periyodik gezegen geçişi (transit) düşüşleri
        + ani parlamalar (flare) + mikro dalgalanmalar.
        """
        n = self.cfg.n_samples
        t = np.linspace(0.0, 1.0, n)

        base = (0.55
                + 0.18 * np.sin(2 * np.pi * 3.0 * t)
                + 0.10 * np.sin(2 * np.pi * 7.3 * t + 0.8))

        # 5 adet transit: her periyodun ortasında Gauss biçimli parlaklık düşüşü
        phase = (t * 5.0) % 1.0
        transit = -0.30 * np.exp(-((phase - 0.5) / 0.05) ** 2)

        # Rastgele konumlarda ani parlama + yavaş sönüm
        flares = np.zeros(n)
        for t0 in self.rng.uniform(0.1, 0.9, size=3):
            decay = np.where(t >= t0, np.exp(-(t - t0) / 0.015), 0.0)
            flares += 0.25 * decay

        # Mikro iniş çıkışlar (hafifçe yumuşatılmış gürültü)
        micro = SignalTools.moving_average(
            self.rng.normal(0.0, 0.05, n), window=5)

        return SignalTools.min_max_normalize(base + transit + flares + micro)

    def pulsar_radio(self, period_samples: int = 8) -> np.ndarray:
        """ATNF benzeri pulsar sinyali: düzenli, keskin radyo vuruşları.

        Her 4. puls vurgulu (güçlü), diğerleri daha zayıftır; gerçek
        pulsarlardaki 'nulling' olayını taklit etmek için nadiren puls atlanır.
        """
        n = self.cfg.n_samples
        x = np.arange(n, dtype=float)
        signal = np.zeros(n)

        for k, center in enumerate(range(0, n, period_samples)):
            accent = (k % 4 == 0)
            if not accent and self.rng.random() < 0.07:   # nulling
                continue
            amplitude = 1.0 if accent else self.rng.uniform(0.45, 0.70)
            pos = center + self.rng.normal(0.0, 0.15)     # küçük zamanlama sapması
            signal += amplitude * np.exp(-0.5 * ((x - pos) / 0.7) ** 2)

        signal += self.rng.normal(0.0, 0.03, n)
        signal = np.clip(signal, 0.0, None)
        return signal / signal.max()

    def gravitational_wave(self) -> np.ndarray:
        """LIGO benzeri kütleçekim dalgası verisi (0-1 arası).

        Düşük frekanslı yavaş salınımlar + birleşme öncesi yükselen 'chirp'
        + birleşme anında ani çöküş ve sönümlü 'ringdown'.
        """
        n = self.cfg.n_samples
        t = np.linspace(0.0, 1.0, n)
        merger = 0.62

        slow = 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.8 * np.sin(2 * np.pi * 0.6 * t + 1.0)

        # İnspiral: genlik ve frekans birleşmeye doğru artar
        inspiral_env = np.where(t < merger, np.exp((t - merger) * 6.0), 0.0)
        chirp = inspiral_env * np.sin(2 * np.pi * (3.0 * t + 25.0 * t ** 3))

        # Ringdown: birleşme sonrası sönümlü salınım
        ringdown_env = np.where(t >= merger, np.exp(-(t - merger) * 18.0), 0.0)
        ringdown = 1.2 * ringdown_env * np.sin(2 * np.pi * 14.0 * (t - merger))

        # Birleşme anında ani seviye çöküşü
        collapse = -1.5 / (1.0 + np.exp(-(t - merger) * 200.0))

        noise = self.rng.normal(0.0, 0.03, n)
        return SignalTools.min_max_normalize(
            slow + chirp + ringdown + collapse + noise)

    def generate_all(self) -> Dict[str, np.ndarray]:
        """Üç veri setini sözlük olarak döndürür."""
        return {
            "light_curve": self.stellar_light_curve(),
            "pulsar": self.pulsar_radio(),
            "gravity": self.gravitational_wave(),
        }


# --------------------------------------------------------------------------
# 2. Müzik teorisi katmanı (akıllı eşleme)
# --------------------------------------------------------------------------
class ScaleMapper:
    """Sayısal değerleri belirli bir oktav aralığındaki skala notalarına eşler."""

    def __init__(self, scale_name: str, low: int, high: int) -> None:
        intervals = SCALES[scale_name]
        # Nota havuzu: aralıktaki yalnızca skalaya ait MIDI notaları
        self.pool = np.array(
            [n for n in range(low, high + 1) if n % 12 in intervals])
        self.low, self.high = low, high

    def map_normalized(self, values: np.ndarray) -> np.ndarray:
        """[0, 1] değerleri nota havuzu indekslerine eşler (Snap to Scale)."""
        idx = np.rint(np.clip(values, 0, 1) * (len(self.pool) - 1)).astype(int)
        return self.pool[idx]

    def snap(self, note: int) -> int:
        """Herhangi bir MIDI notasını havuzdaki en yakın notaya yuvarlar."""
        return int(self.pool[np.argmin(np.abs(self.pool - note))])

    def describe(self) -> str:
        return " ".join(midi_to_name(int(n)) for n in self.pool)


def merge_repeats(pitches: np.ndarray, velocities: np.ndarray,
                  step: float, max_steps: int) -> List[Tuple[float, float, int, int]]:
    """Ardışık aynı perdeleri tek uzun notada birleştirir.

    Döndürür: (başlangıç_vuruşu, süre_vuruş, perde, velocity) listesi.
    """
    merged: List[Tuple[float, float, int, int]] = []
    i = 0
    while i < len(pitches):
        j = i + 1
        while j < len(pitches) and pitches[j] == pitches[i] and (j - i) < max_steps:
            j += 1
        merged.append((i * step, (j - i) * step, int(pitches[i]), int(velocities[i])))
        i = j
    return merged


# --------------------------------------------------------------------------
# 3. Enstrüman rolleri ve algoritmik eşleme
# --------------------------------------------------------------------------
class MelodySonifier:
    """TRACK 1: Yıldız parlaklığı -> C4-C6 melodisi (parlaklık artar, perde yükselir)."""

    STEP_BEATS = 0.5   # 8'lik nota çözünürlüğü

    def __init__(self, cfg: SymphonyConfig, mapper: ScaleMapper) -> None:
        self.cfg, self.mapper = cfg, mapper

    def generate(self, light_curve: np.ndarray) -> List[NoteEvent]:
        n_steps = int(self.cfg.total_beats / self.STEP_BEATS)

        norm = SignalTools.min_max_normalize(light_curve)
        log("1/4 NORMALİZASYON", f"Işık eğrisi [{light_curve.min():.2f}, "
            f"{light_curve.max():.2f}] -> [0, 1] (Min-Max)")

        grid = SignalTools.resample_mean(norm, n_steps)
        grid = SignalTools.min_max_normalize(SignalTools.moving_average(grid, 3))

        pitches = self.mapper.map_normalized(grid)
        log("2/4 NOTAYA DÖNÜŞÜM", f"Melodi: {n_steps} adım -> "
            f"{midi_to_name(int(pitches.min()))}-{midi_to_name(int(pitches.max()))} "
            f"aralığında skala notalarına eşlendi")

        base_vel = 55 + 35 * grid
        merged = merge_repeats(pitches, base_vel, self.STEP_BEATS, max_steps=6)

        events = [
            NoteEvent(start, dur * 0.95, pitch, to_velocity(vel, start, self.cfg), CH_MELODY)
            for start, dur, pitch, vel in merged
        ]
        log("2/4 NOTAYA DÖNÜŞÜM", f"TRACK 1 (Yıldız Melodisi): {len(events)} nota")
        return events


class PulsarSonifier:
    """TRACK 2: Pulsar tepe noktaları -> C2 kick / hi-hat ritmi."""

    GRID_BEATS = 0.25          # 16'lık nota ızgarası
    HIT_DURATION = 0.10
    ACCENT_THRESHOLD = 0.85    # güçlü pulslar kick, diğerleri hi-hat

    def __init__(self, cfg: SymphonyConfig) -> None:
        self.cfg = cfg

    def generate(self, pulsar: np.ndarray) -> List[NoteEvent]:
        threshold = pulsar.mean() + 0.5 * pulsar.std()
        peaks = SignalTools.find_peaks(pulsar, threshold)
        log("1/4 NORMALİZASYON", f"Pulsar: eşik={threshold:.3f}, "
            f"{len(peaks)} tepe noktası (peak) algılandı")

        samples_per_beat = len(pulsar) / self.cfg.total_beats
        strongest = pulsar[peaks].max() if len(peaks) else 1.0

        # Tepe noktalarını 16'lık ızgaraya oturt; aynı konuma düşenlerden güçlüsünü tut
        hits: Dict[float, float] = {}
        for idx in peaks:
            beat = round((idx / samples_per_beat) / self.GRID_BEATS) * self.GRID_BEATS
            strength = float(pulsar[idx] / strongest)
            hits[beat] = max(strength, hits.get(beat, 0.0))

        events: List[NoteEvent] = []
        for beat, strength in sorted(hits.items()):
            if beat >= self.cfg.total_beats:
                continue
            is_kick = strength >= self.ACCENT_THRESHOLD
            pitch = KICK_NOTE if is_kick else HIHAT_NOTE
            velocity = to_velocity(55 + 60 * strength, beat, self.cfg)
            events.append(NoteEvent(beat, self.HIT_DURATION, pitch, velocity, CH_DRUMS))

        kicks = sum(e.pitch == KICK_NOTE for e in events)
        log("2/4 NOTAYA DÖNÜŞÜM", f"TRACK 2 (Pulsar Ritmi): {len(events)} vuruş "
            f"({kicks} kick / {len(events) - kicks} hi-hat)")
        return events


class BassPadSonifier:
    """TRACK 3: Kütleçekim dalgası -> C1-C3 aralığında uzun bas / pad dokusu."""

    def __init__(self, cfg: SymphonyConfig, mapper: ScaleMapper) -> None:
        self.cfg, self.mapper = cfg, mapper

    def generate(self, gravity: np.ndarray) -> List[NoteEvent]:
        step = float(self.cfg.beats_per_bar)       # ölçü başına bir nota
        n_steps = self.cfg.bars

        norm = SignalTools.min_max_normalize(gravity)
        grid = SignalTools.min_max_normalize(SignalTools.resample_mean(norm, n_steps))
        pitches = self.mapper.map_normalized(grid)
        log("2/4 NOTAYA DÖNÜŞÜM", f"Bas/Pad: {n_steps} ölçü -> "
            f"{midi_to_name(int(pitches.min()))}-{midi_to_name(int(pitches.max()))} "
            f"derin oktavlarına eşlendi")

        base_vel = 70 + 25 * grid
        merged = merge_repeats(pitches, base_vel, step, max_steps=4)

        events: List[NoteEvent] = []
        for start, dur, pitch, vel in merged:
            velocity = to_velocity(vel, start, self.cfg)
            events.append(NoteEvent(start, dur, pitch, velocity, CH_BASS))
            # Pad dokusu: kök notanın üstüne skalaya snap edilmiş bir 'beşli'
            fifth = self.mapper.snap(pitch + 7)
            if fifth != pitch:
                events.append(NoteEvent(start, dur, fifth,
                                        max(1, velocity - 15), CH_BASS))
        log("2/4 NOTAYA DÖNÜŞÜM", f"TRACK 3 (Kara Delik Atmosferi): {len(events)} uzun nota")
        return events


# --------------------------------------------------------------------------
# 4. Çıktı katmanı: MIDI birleştirme ve dışa aktarma
# --------------------------------------------------------------------------
class MidiComposer:
    """Üç kanalı tek bir MIDI (type 1) dosyasında birleştirir."""

    def __init__(self, cfg: SymphonyConfig) -> None:
        self.cfg = cfg

    def _build_track(self, name: str, program: int, channel: int,
                     events: Sequence[NoteEvent],
                     with_tempo: bool = False) -> "mido.MidiTrack":
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if with_tempo:
            track.append(mido.MetaMessage(
                "set_tempo", tempo=mido.bpm2tempo(self.cfg.bpm), time=0))
            track.append(mido.MetaMessage(
                "time_signature", numerator=self.cfg.beats_per_bar,
                denominator=4, time=0))
            track.append(mido.MetaMessage("key_signature", key="Cm", time=0))
        if channel != CH_DRUMS:   # perküsyon kanalında program change gerekmez
            track.append(mido.Message(
                "program_change", program=program, channel=channel, time=0))

        # Mutlak tick'lere çevir; aynı tick'te note_off, note_on'dan önce gelsin
        tpb = self.cfg.ticks_per_beat
        raw = []
        for e in events:
            on = int(round(e.start_beat * tpb))
            off = on + max(1, int(round(e.duration_beats * tpb)))
            raw.append((on, 1, mido.Message("note_on", note=e.pitch,
                                            velocity=e.velocity, channel=e.channel)))
            raw.append((off, 0, mido.Message("note_off", note=e.pitch,
                                             velocity=0, channel=e.channel)))
        raw.sort(key=lambda item: (item[0], item[1]))

        last_tick = 0
        for tick, _, msg in raw:
            track.append(msg.copy(time=tick - last_tick))
            last_tick = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        return track

    def compose(self, melody: Sequence[NoteEvent], drums: Sequence[NoteEvent],
                bass: Sequence[NoteEvent]) -> "mido.MidiFile":
        midi = mido.MidiFile(type=1, ticks_per_beat=self.cfg.ticks_per_beat)
        # Program numaraları (General MIDI, 0 tabanlı): 0 = Akustik Piyano,
        # 89 = Warm Pad. Pluck sesi için 46 (Orchestral Harp) denenebilir.
        midi.tracks.append(self._build_track(
            "Track 1 - Yildiz Melodisi (Piyano)", 0, CH_MELODY, melody, with_tempo=True))
        midi.tracks.append(self._build_track(
            "Track 2 - Pulsar Ritmi (Perkusyon)", 0, CH_DRUMS, drums))
        midi.tracks.append(self._build_track(
            "Track 3 - Kara Delik Atmosferi (Bas/Pad)", 89, CH_BASS, bass))
        return midi

    def export(self, midi: "mido.MidiFile") -> Path:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.output_dir / self.cfg.midi_name
        midi.save(str(path))
        return path


# --------------------------------------------------------------------------
# Görselleştirme (3 panelli matplotlib grafiği)
# --------------------------------------------------------------------------
class SymphonyVisualizer:
    """Ham veri ile nota/frekans eşleşmesini (Pitch vs. Time) görselleştirir."""

    COLORS = {"light": "#ffd166", "pulsar": "#ef476f", "gravity": "#4cc9f0",
              "kick": "#ff595e", "hat": "#ffca3a"}

    def __init__(self, cfg: SymphonyConfig) -> None:
        self.cfg = cfg

    def _seconds(self, events: Sequence[NoteEvent]):
        starts = [self.cfg.beat_to_seconds(e.start_beat) for e in events]
        ends = [self.cfg.beat_to_seconds(e.start_beat + e.duration_beats) for e in events]
        return np.array(starts), np.array(ends)

    def plot(self, data: Dict[str, np.ndarray], melody: Sequence[NoteEvent],
             drums: Sequence[NoteEvent], bass: Sequence[NoteEvent],
             pitch_pool: Sequence[int]) -> Path:
        c = self.COLORS
        t_raw = np.linspace(0, self.cfg.duration_seconds, self.cfg.n_samples)

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
            fig.suptitle("Samanyolu Senfonisi - Uzay Verisinden Müziğe",
                         fontsize=16, fontweight="bold")

            # Panel 1: ham uzay verileri (normalize, üst üste ofsetli)
            ax = axes[0]
            series = [("Işık eğrisi (TESS/Kepler)", data["light_curve"], 2.3, c["light"]),
                      ("Pulsar radyo (ATNF)", data["pulsar"], 1.15, c["pulsar"]),
                      ("Kütleçekim dalgası (LIGO)", data["gravity"], 0.0, c["gravity"])]
            for label, values, offset, color in series:
                ax.plot(t_raw, SignalTools.min_max_normalize(values) + offset,
                        color=color, lw=0.9, label=label)
            ax.set_yticks([])
            ax.set_title("Panel 1 - Ham uzay verisi (normalize edilmiş)")
            ax.legend(loc="upper right", fontsize=8, ncol=3)

            # Panel 2: piyano-roll (MIDI perdesi - zaman)
            ax = axes[1]
            for events, color in ((melody, c["light"]), (bass, c["gravity"])):
                s, e = self._seconds(events)
                ax.hlines([ev.pitch for ev in events], s, e, color=color, lw=3)
            for kind, note, color in (("Kick", KICK_NOTE, c["kick"]),
                                      ("Hi-hat", HIHAT_NOTE, c["hat"])):
                hits = [ev for ev in drums if ev.pitch == note]
                s, _ = self._seconds(hits)
                ax.scatter(s, [note] * len(s), marker="|", s=60, color=color, label=kind)
            ticks = sorted(set(int(p) for p in pitch_pool) | {HIHAT_NOTE})
            ax.set_yticks(ticks)
            ax.set_yticklabels([midi_to_name(t) for t in ticks], fontsize=6)
            ax.set_ylabel("Nota (perde)")
            ax.set_title("Panel 2 - Pitch vs. Time: melodi, bas/pad ve pulsar vuruşları")
            ax.legend(loc="upper right", fontsize=8)

            # Panel 3: frekans (Hz, logaritmik) - zaman
            ax = axes[2]
            for events, color in ((melody, c["light"]), (bass, c["gravity"])):
                s, e = self._seconds(events)
                freqs = [midi_to_freq(ev.pitch) for ev in events]
                ax.hlines(freqs, s, e, color=color, lw=3)
            s, _ = self._seconds(drums)
            ax.scatter(s, [30] * len(s), marker="|", s=40, color=c["pulsar"])
            ax.set_yscale("log")
            ax.set_ylim(27, 1400)
            ax.set_ylabel("Frekans (Hz)")
            ax.set_xlabel("Zaman (saniye)")
            ax.set_title("Panel 3 - Notaların karşılık geldiği frekanslar")
            ax.grid(alpha=0.2, which="both")

            fig.tight_layout(rect=(0, 0, 1, 0.97))
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.cfg.output_dir / self.cfg.figure_name
            fig.savefig(path, dpi=150)

        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)
        return path


# --------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------
class MilkyWaySymphony:
    """Tüm katmanları sırayla çalıştıran ana sınıf."""

    def __init__(self, cfg: SymphonyConfig | None = None) -> None:
        self.cfg = cfg or SymphonyConfig()

    def run(self, data: Dict[str, np.ndarray] | None = None) -> Path:
        cfg = self.cfg
        log("BAŞLANGIÇ", f"Samanyolu Senfonisi | {cfg.bpm} BPM | "
            f"{cfg.bars} ölçü (~{cfg.duration_seconds:.0f} sn) | skala: {cfg.scale}")

        # Giriş katmanı
        if data is None:
            log("0/4 VERİ", "Mock (yapay) uzay verileri üretiliyor...")
            data = MockDataGenerator(cfg).generate_all()
        for name, values in data.items():
            log("0/4 VERİ", f"{name}: {len(values)} örnek, "
                f"min={values.min():.3f}, max={values.max():.3f}")

        # Müzik teorisi katmanı
        melody_mapper = ScaleMapper(cfg.scale, *cfg.melody_range)
        bass_mapper = ScaleMapper(cfg.scale, *cfg.bass_range)
        log("MÜZİK TEORİSİ", f"Melodi nota havuzu: {melody_mapper.describe()}")
        log("MÜZİK TEORİSİ", f"Bas nota havuzu  : {bass_mapper.describe()}")

        # Enstrüman rolleri
        melody = MelodySonifier(cfg, melody_mapper).generate(data["light_curve"])
        drums = PulsarSonifier(cfg).generate(data["pulsar"])
        bass = BassPadSonifier(cfg, bass_mapper).generate(data["gravity"])

        # Çıktı katmanı
        composer = MidiComposer(cfg)
        midi = composer.compose(melody, drums, bass)
        log("3/4 KANAL BİRLEŞTİRME", f"{len(midi.tracks)} MIDI track tek dosyada birleştirildi "
            f"(toplam {len(melody) + len(drums) + len(bass)} nota olayı)")
        midi_path = composer.export(midi)
        log("4/4 DIŞA AKTARIM", f"MIDI dosyası kaydedildi: {midi_path}")

        pool = list(melody_mapper.pool) + list(bass_mapper.pool)
        fig_path = SymphonyVisualizer(cfg).plot(data, melody, drums, bass, pool)
        log("4/4 DIŞA AKTARIM", f"Grafik kaydedildi: {fig_path}")
        log("BİTTİ", "Samanyolu Senfonisi hazır. İyi dinlemeler!")
        return midi_path


def main() -> None:
    MilkyWaySymphony(SymphonyConfig(bpm=60, scale="minor_pentatonic")).run()


if __name__ == "__main__":
    main()