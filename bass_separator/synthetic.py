"""Material sintético para pruebas y demostraciones.

Genera una línea de walking bass, un "conjunto" de acompañamiento (piano, batería,
metales) y una degradación tipo 78 rpm. No pretende sonar como una big band real;
sirve para validar el seguimiento de altura, el realce y el pipeline sin
necesitar grabaciones con derechos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal

from .dsp import bandpass, highpass, lowpass, midi_to_hz, normalize_peak
from .notes import Note
from .synth import DoubleBassSynth


# ---------------------------------------------------------------------------
# Línea de walking bass
# ---------------------------------------------------------------------------
_II_V_I_C = [  # progresión jazz sencilla (acordes por compás): raíz midi, tipo
    (50, "m7"), (43, "7"), (48, "maj7"), (48, "maj7"),
    (45, "m7"), (50, "7"), (43, "maj7"), (43, "maj7"),
]
_CHORD_TONES = {"m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11]}


def walking_bass_line(n_bars: int = 8, bpm: float = 160.0, seed: int = 0,
                      lowest: int = 28, highest: int = 55) -> List[Note]:
    """Negras de walking bass sobre una progresión II-V-I (E1=28 .. G3=55)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    notes: List[Note] = []
    t = 0.0
    prev = None
    for bar in range(n_bars):
        root, kind = _II_V_I_C[bar % len(_II_V_I_C)]
        tones = [root + iv for iv in _CHORD_TONES[kind]]
        next_root = _II_V_I_C[(bar + 1) % len(_II_V_I_C)][0]
        for b in range(4):
            if b == 0:
                m = root
            elif b == 3:
                m = next_root + int(rng.choice([-1, 1, -2, 2]))  # aproximación cromática
            else:
                m = int(rng.choice(tones)) + int(rng.choice([0, 0, 12, -12]))
            # mantener en tesitura y evitar saltos enormes
            while m < lowest:
                m += 12
            while m > highest:
                m -= 12
            if prev is not None and abs(m - prev) > 9 and rng.random() < 0.7:
                m += 12 if m < prev else -12
                m = int(np.clip(m, lowest, highest))
            dur = beat * rng.uniform(0.85, 0.98)
            vel = int(rng.integers(80, 115))
            notes.append(Note(start=t, end=t + dur, midi=m, f0=float(midi_to_hz(m)),
                              velocity=vel, confidence=1.0))
            prev = m
            t += beat
    return notes


def render_bass(notes: List[Note], sr: int, duration: float, seed: int = 0) -> np.ndarray:
    return DoubleBassSynth(sr, brightness=0.5, seed=seed).render(notes, duration)


# ---------------------------------------------------------------------------
# Acompañamiento sintético
# ---------------------------------------------------------------------------
def _tone(f0: float, dur: float, sr: int, harmonics: int = 8, decay: float = 1.5,
          rolloff: float = 1.2) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(1, harmonics + 1):
        y += np.sin(2 * np.pi * f0 * k * t) / k ** rolloff
    env = np.exp(-t * decay) * np.minimum(1.0, t * 200)
    return (y * env).astype(np.float32)


def render_band(duration: float, sr: int, bpm: float = 160.0, seed: int = 1,
                with_piano: bool = True, with_drums: bool = True, with_horns: bool = True) -> np.ndarray:
    """Piano (acordes), batería (bombo/caja/charles) y metales (melodía sostenida)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    n_beats = int(duration / beat) + 1

    if with_piano:
        for i in range(n_beats):
            if i % 2 == 1 or rng.random() < 0.3:
                root, kind = _II_V_I_C[(i // 4) % len(_II_V_I_C)]
                for iv in _CHORD_TONES[kind]:
                    f = float(midi_to_hz(root + 24 + iv))
                    tone = _tone(f, beat * 0.9, sr, harmonics=6, decay=3.0) * 0.12
                    a = int(i * beat * sr + rng.uniform(0, 0.01) * sr)
                    b = min(n, a + len(tone))
                    if b > a:
                        out[a:b] += tone[:b - a]

    if with_drums:
        for i in range(n_beats):
            a = int(i * beat * sr)
            # bombo en 1 y 3 (ligero), caja en 2 y 4, charles en todas
            if a >= n:
                break
            if i % 2 == 0:
                t = np.arange(int(0.25 * sr)) / sr
                kick = np.sin(2 * np.pi * (55 + 40 * np.exp(-t * 30)) * t) * np.exp(-t * 18) * 0.25
                b = min(n, a + len(kick))
                out[a:b] += kick[:b - a].astype(np.float32)
            else:
                t = np.arange(int(0.18 * sr)) / sr
                snare = (rng.normal(0, 1, len(t)) * np.exp(-t * 25) * 0.2
                         + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.15)
                b = min(n, a + len(snare))
                out[a:b] += snare[:b - a].astype(np.float32)
            t = np.arange(int(0.08 * sr)) / sr
            hat = rng.normal(0, 1, len(t)) * np.exp(-t * 60) * 0.08
            hat = highpass(hat.astype(np.float32), sr, 5000.0, order=2)
            b = min(n, a + len(hat))
            out[a:b] += hat[:b - a]

    if with_horns:
        melody = [67, 70, 72, 74, 72, 70, 67, 65, 62, 65, 67, 70]
        t0 = 0.0
        k = 0
        while t0 < duration - beat:
            dur = beat * rng.choice([1, 2, 3]) * 0.95
            m = melody[k % len(melody)]
            k += 1
            nn = int(dur * sr)
            t = np.arange(nn) / sr
            f = float(midi_to_hz(m)) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
            ph = 2 * np.pi * np.cumsum(f) / sr
            saw = sum(np.sin(h * ph) / h for h in range(1, 12))
            env = np.minimum(1, t * 40) * np.minimum(1, (dur - t) * 40)
            tone = (saw * env * 0.09).astype(np.float32)
            a = int(t0 * sr)
            b = min(n, a + nn)
            if b > a:
                out[a:b] += tone[:b - a]
            t0 += dur / 0.95
    return out


# ---------------------------------------------------------------------------
# Degradación tipo 78 rpm
# ---------------------------------------------------------------------------
def degrade_78rpm(y: np.ndarray, sr: int, noise_db: float = -38.0, crackle_rate: float = 25.0,
                  lo: float = 80.0, hi: float = 4500.0, seed: int = 2) -> np.ndarray:
    """Limita el ancho de banda, añade ruido de superficie, crepitar y saturación suave."""
    rng = np.random.default_rng(seed)
    y = y.astype(np.float32)
    mono = y if y.ndim == 1 else y.mean(axis=0)
    out = bandpass(mono, sr, lo, hi, order=2)
    n = len(out)
    # ruido de superficie (rosa aproximado)
    white = rng.normal(0, 1, n).astype(np.float32)
    pink = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408],
                          [1, -2.494956, 2.017265, -0.522189], white).astype(np.float32)
    pink = bandpass(pink, sr, 60.0, 6000.0, order=1)
    pink *= 10 ** (noise_db / 20) / (np.sqrt(np.mean(pink ** 2)) + 1e-9)
    out = out + pink
    # crepitar: impulsos aleatorios
    n_clicks = int(crackle_rate * n / sr)
    pos = rng.integers(0, n - 4, n_clicks)
    amp = rng.uniform(0.05, 0.5, n_clicks) * rng.choice([-1, 1], n_clicks)
    for p, a in zip(pos, amp):
        out[p:p + 3] += a * np.array([1.0, -0.6, 0.2], dtype=np.float32)
    # saturación suave
    out = np.tanh(out * 1.3) / 1.3
    return normalize_peak(out.astype(np.float32), -1.0)


@dataclass
class SyntheticSong:
    mix: np.ndarray        # mezcla degradada (mono)
    bass: np.ndarray       # contrabajo limpio (referencia)
    band: np.ndarray       # resto de instrumentos (limpios)
    notes: List[Note]
    sr: int


def make_song(duration: float = 12.0, sr: int = 44100, bpm: float = 160.0, seed: int = 0,
              degrade: bool = True, bass_db: float = -6.0) -> SyntheticSong:
    """Canción sintética completa con referencia de contrabajo."""
    n_bars = int(np.ceil(duration * bpm / 60.0 / 4)) + 1
    notes = [nt for nt in walking_bass_line(n_bars, bpm, seed) if nt.end <= duration]
    bass = render_bass(notes, sr, duration, seed)
    band = render_band(duration, sr, bpm, seed + 1)
    n = min(len(bass), len(band))
    bass, band = bass[:n], band[:n]
    bass = normalize_peak(bass, -3.0) * 10 ** (bass_db / 20)
    band = normalize_peak(band, -3.0)
    mix = bass + band
    if degrade:
        mix = degrade_78rpm(mix, sr, seed=seed + 2)
    else:
        mix = normalize_peak(mix, -1.0)
    return SyntheticSong(mix=mix, bass=bass, band=band, notes=notes, sr=sr)
