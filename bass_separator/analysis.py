"""Análisis previo de la grabación y elección automática de parámetros.

Se ejecuta sobre el audio original, antes de separar, y mide:

* mono efectivo, ancho de banda, nivel, relación señal/ruido, crepitar, zumbido;
* desviación de afinación respecto a A = 440 Hz (en cents) -> corrección de velocidad;
* tempo y rejilla de pulsos -> priores rítmicos para la segmentación de notas;
* tonalidad -> prior de alturas diatónicas para el seguidor de f0;
* una puntuación "vintage" que decide el perfil y la estrategia de separación
  (ensamble de modelos y aumentos en tiempo de prueba) según la calidad pedida y
  el dispositivo disponible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .audio_io import is_effectively_mono, resample, to_mono
from .config import PRESETS, Settings, apply_preset
from .dsp import amp2db, bandpass, rms
from .restore import declick, detect_hum

ANALYSIS_SR = 22050
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# perfiles de Krumhansl-Kessler
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SCALE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
_MINOR_SCALE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)   # natural + sensible


def _py(v):
    """Convierte escalares de numpy a tipos nativos (para JSON)."""
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 3)
    return v


@dataclass
class AudioAnalysis:
    sr: int
    duration_s: float
    channels: int
    mono: bool
    rms_db: float
    peak_db: float
    band_lo_hz: float
    band_hi_hz: float
    low_end_db: float            # energía 40-100 Hz relativa a 100-300 Hz
    snr_db: float
    crackle_per_s: float
    hum_hz: float
    tuning_cents: float          # + = la grabación suena aguda respecto a A440
    tuning_confidence: float     # 0..1
    tempo_bpm: float
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    key: str = "?"
    key_confidence: float = 0.0
    key_pcs: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=bool))
    vintage_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beats"] = len(self.beats)
        d["key_pcs"] = [KEY_NAMES[i] for i in range(12) if self.key_pcs[i]]
        return {k: _py(v) for k, v in d.items()}

    def report(self) -> str:
        lines = [
            f"Duración {self.duration_s:.1f} s, {self.channels} canal(es) a {self.sr} Hz, "
            f"{'mono efectivo' if self.mono else 'estéreo real'}",
            f"Nivel RMS {self.rms_db:.1f} dBFS, pico {self.peak_db:.1f} dBFS, SNR estimada {self.snr_db:.0f} dB",
            f"Ancho de banda ~{self.band_lo_hz:.0f}–{self.band_hi_hz:.0f} Hz, graves 40-100 Hz {self.low_end_db:+.1f} dB "
            f"respecto a 100-300 Hz",
            f"Crepitar {self.crackle_per_s:.1f} clics/s, zumbido {'%.0f Hz' % self.hum_hz if self.hum_hz else 'no'}",
            f"Afinación {self.tuning_cents:+.0f} cents respecto a A440 (confianza {self.tuning_confidence:.2f})",
            f"Tempo ~{self.tempo_bpm:.0f} BPM ({len(self.beats)} pulsos), tonalidad {self.key} "
            f"(confianza {self.key_confidence:.2f})",
            f"Puntuación vintage {self.vintage_score:.2f} -> perfil {'vintage' if self.vintage_score >= 0.5 else 'modern'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Medidas individuales
# ---------------------------------------------------------------------------
def estimate_tuning_cents(y: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 3000.0) -> Tuple[float, float]:
    """Desviación de afinación (cents, en [-50, 50)) y confianza 0..1.

    Se recogen los picos espectrales estables (piptrack) y se construye un
    histograma circular de su resto respecto a la rejilla temperada de A440.
    """
    mono = to_mono(y)
    if sr != ANALYSIS_SR:
        mono = resample(mono, sr, ANALYSIS_SR)
    sr_a = ANALYSIS_SR
    pitches, mags = librosa.piptrack(y=mono, sr=sr_a, n_fft=4096, hop_length=1024, fmin=fmin, fmax=fmax,
                                     threshold=0.2)
    sel = (pitches > 0) & (mags > 0)
    if sel.sum() < 50:
        return 0.0, 0.0
    f = pitches[sel]
    w = np.log1p(mags[sel] / (np.median(mags[sel]) + 1e-12))
    cents = 1200.0 * np.log2(f / 440.0)
    resid = ((cents + 50.0) % 100.0) - 50.0
    hist, _ = np.histogram(resid, bins=100, range=(-50, 50), weights=w)
    # suavizado circular
    hist = gaussian_filter1d(np.concatenate([hist, hist, hist]), sigma=2.5)[100:200]
    k = int(np.argmax(hist))
    a, b, c = hist[(k - 1) % 100], hist[k], hist[(k + 1) % 100]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
    est = -50.0 + 0.5 + k + delta
    est = ((est + 50.0) % 100.0) - 50.0
    conf = float(np.clip((b - hist.mean()) / (b + 1e-12), 0.0, 1.0))
    return float(est), conf


def estimate_key(y: np.ndarray, sr: int) -> Tuple[str, float, np.ndarray]:
    """Tonalidad (Krumhansl-Kessler) y máscara de clases de altura diatónicas."""
    mono = to_mono(y)
    if sr != ANALYSIS_SR:
        mono = resample(mono, sr, ANALYSIS_SR)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=ANALYSIS_SR, hop_length=2048, fmin=librosa.note_to_hz("C2"))
    prof = chroma.mean(axis=1)
    if prof.sum() <= 0:
        return "?", 0.0, np.ones(12, dtype=bool)
    scores = []
    for tonic in range(12):
        for name, template, scale in (("major", _MAJOR, _MAJOR_SCALE), ("minor", _MINOR, _MINOR_SCALE)):
            t = np.roll(template, tonic)
            r = np.corrcoef(prof, t)[0, 1]
            scores.append((r, tonic, name, np.roll(scale, tonic)))
    scores.sort(key=lambda s: -s[0])
    best, second = scores[0], scores[1]
    conf = float(np.clip((best[0] - second[0]) * 4.0 + 0.2, 0.0, 1.0))
    return f"{KEY_NAMES[best[1]]} {best[2]}", conf, best[3]


def estimate_tempo(y: np.ndarray, sr: int, start_bpm: float = 160.0) -> Tuple[float, np.ndarray]:
    mono = to_mono(y)
    if sr != ANALYSIS_SR:
        mono = resample(mono, sr, ANALYSIS_SR)
    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=ANALYSIS_SR, hop_length=512, start_bpm=start_bpm,
                                               units="time")
    except Exception:  # pragma: no cover - señales degeneradas
        return 0.0, np.zeros(0)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return tempo, np.asarray(beats, dtype=float)


def estimate_snr(y: np.ndarray, sr: int) -> float:
    """SNR aproximada: percentil 90 frente a percentil 10 de la energía espectral por bin."""
    mono = to_mono(y)
    if sr != ANALYSIS_SR:
        mono = resample(mono, sr, ANALYSIS_SR)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=1024)) ** 2
    noise = np.percentile(S, 10, axis=1)
    sig = np.percentile(S, 90, axis=1)
    return float(10 * np.log10(sig.sum() / (noise.sum() + 1e-20) + 1e-20))


def estimate_bandwidth(y: np.ndarray, sr: int, rel_db: float = -40.0) -> Tuple[float, float, float]:
    mono = to_mono(y)
    f, p = signal.welch(mono, fs=sr, nperseg=1 << 14)
    pdb = 10 * np.log10(p + 1e-20)
    peak = pdb.max()
    above = f[pdb > peak + rel_db]
    lo, hi = (float(above.min()), float(above.max())) if len(above) else (0.0, sr / 2)
    low = pdb[(f >= 40) & (f < 100)].mean()
    mid = pdb[(f >= 100) & (f < 300)].mean()
    return lo, hi, float(low - mid)


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------
def analyze_audio(y: np.ndarray, sr: int, max_seconds: float = 240.0) -> AudioAnalysis:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    n = y.shape[-1]
    excerpt = y[..., : int(max_seconds * sr)]
    mono = to_mono(excerpt)
    lo, hi, low_end = estimate_bandwidth(excerpt, sr)
    snr = estimate_snr(excerpt, sr)
    _, n_clicks = declick(mono[: int(60 * sr)], sr)
    crackle = n_clicks / max(1.0, min(60.0, len(mono) / sr))
    hum = detect_hum(excerpt, sr)
    cents, conf = estimate_tuning_cents(excerpt, sr)
    tempo, beats = estimate_tempo(y, sr)           # pulsos de toda la pieza
    key, kconf, pcs = estimate_key(excerpt, sr)
    mono_flag = bool(is_effectively_mono(y))
    vintage = (0.3 * float(mono_flag) + 0.3 * float(hi < 8000) + 0.2 * float(snr < 30)
               + 0.2 * float(crackle > 0.3))
    return AudioAnalysis(sr=sr, duration_s=n / sr, channels=y.shape[0], mono=mono_flag,
                         rms_db=float(amp2db(rms(mono))), peak_db=float(amp2db(np.abs(mono).max() + 1e-12)),
                         band_lo_hz=lo, band_hi_hz=hi, low_end_db=low_end, snr_db=snr,
                         crackle_per_s=crackle, hum_hz=hum, tuning_cents=cents, tuning_confidence=conf,
                         tempo_bpm=tempo, beats=beats, key=key, key_confidence=kconf, key_pcs=pcs,
                         vintage_score=vintage)


# ---------------------------------------------------------------------------
# Recomendación de parámetros
# ---------------------------------------------------------------------------
def crepe_available() -> bool:
    try:
        import torchcrepe  # noqa: F401
        return True
    except Exception:
        return False


def parse_speed_correction(value) -> Optional[float]:
    """'auto' -> None, 'off' -> 0.0, número -> cents."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip().lower()
    if v in ("auto", ""):
        return None
    if v in ("off", "no", "none", "0"):
        return 0.0
    return float(v)


def recommend_settings(an: AudioAnalysis, base: Settings, explicit: Iterable[str] = (),
                       device: str = "auto") -> Tuple[Settings, Dict[str, Tuple[object, str]]]:
    """Devuelve ``(settings, decisiones)``; no toca los parámetros listados en ``explicit``."""
    explicit = set(explicit)
    choices: Dict[str, Tuple[object, str]] = {}
    s = base
    vintage = an.vintage_score >= 0.5

    def set_(name, value, reason):
        nonlocal s
        if name in explicit:
            return
        value = _py(value) if not isinstance(value, (str, tuple)) else value
        s = replace(s, **{name: value})
        choices[name] = (value, reason)

    # perfil
    if "profile" not in explicit:
        prof = "vintage" if vintage else "modern"
        keep = {k: getattr(s, k) for k in PRESETS[prof] if k in explicit}
        s = replace(apply_preset(s, prof), **keep)
        choices["profile"] = (prof, f"puntuación vintage {an.vintage_score:.2f}")

    # ensamble y TTA según calidad y dispositivo
    gpu = device in ("cuda", "mps")
    if device == "auto":
        try:
            from .separation import pick_device
            gpu = pick_device("auto") in ("cuda", "mps")
        except Exception:
            gpu = False
    long_track = an.duration_s > 300
    if s.quality == "fast":
        set_("models", ("htdemucs_ft",), "calidad fast")
        set_("shifts", 1, "calidad fast")
    elif s.quality == "best":
        set_("models", ("htdemucs_ft", "mdx_extra"), "calidad best: ensamble de dos arquitecturas")
        set_("shifts", 3, "calidad best: 3 aumentos en tiempo de prueba")
    else:  # balanced
        if gpu:
            set_("models", ("htdemucs_ft", "mdx_extra"), "GPU disponible: ensamble htdemucs_ft + mdx_extra")
            set_("shifts", 2, "GPU disponible: 2 aumentos en tiempo de prueba")
        else:
            set_("models", ("htdemucs_ft",), "CPU: un solo modelo (usa --quality best para ensamble)")
            set_("shifts", 2 if (vintage and not long_track) else 1,
                 "CPU: 2 shifts en grabación vintage corta" if (vintage and not long_track) else "CPU: 1 shift")

    # corrección de velocidad
    if parse_speed_correction(s.speed_correction) is None:
        if an.tuning_confidence >= 0.25 and abs(an.tuning_cents) >= 8.0:
            set_("speed_correction", f"{an.tuning_cents:.1f}",
                 f"afinación {an.tuning_cents:+.0f} cents (confianza {an.tuning_confidence:.2f})")
        else:
            set_("speed_correction", "off",
                 f"desviación {an.tuning_cents:+.0f} cents no significativa o poco fiable")

    # restauración y realce
    if an.snr_db < 20:
        dn, pd = 0.85, 0.6
    elif an.snr_db < 35:
        dn, pd = 0.7, 0.5
    else:
        dn, pd = 0.4, 0.2
    set_("denoise", dn, f"SNR {an.snr_db:.0f} dB")
    set_("pre_denoise", pd, f"SNR {an.snr_db:.0f} dB")
    set_("declick", an.crackle_per_s > 0.3, f"{an.crackle_per_s:.1f} clics/s")
    set_("hum_notch", an.hum_hz > 0, "zumbido detectado" if an.hum_hz else "sin zumbido")
    if an.low_end_db < -12:
        set_("sub_gain", 1.3, f"graves muy débiles ({an.low_end_db:+.0f} dB)")
    elif an.low_end_db < -6:
        set_("sub_gain", 1.0, f"graves débiles ({an.low_end_db:+.0f} dB)")
    else:
        set_("sub_gain", 0.6, f"graves presentes ({an.low_end_db:+.0f} dB)")
    set_("lowpass_hz", float(np.round(np.clip(an.band_hi_hz * 1.1, 3000.0, 9000.0), -1)),
         f"ancho de banda hasta {an.band_hi_hz:.0f} Hz")
    set_("bleed_reduction", 0.7 if vintage else 0.4, "perfil vintage" if vintage else "perfil modern")

    # seguidor de altura: en las pruebas con material degradado, salience + priores
    # supera a la fusión con CREPE, así que CREPE queda como opción explícita.
    if s.pitch_method == "auto":
        set_("pitch_method", "salience", "salience + priores musicales (CREPE solo bajo demanda)")
    if s.crepe_model == "auto":
        set_("crepe_model", "full" if gpu else "tiny", "GPU: CREPE full" if gpu else "CPU: CREPE tiny (rápido)")
    return s, choices


def describe_choices(choices: Dict[str, Tuple[object, str]]) -> str:
    return "\n".join(f"  {k:18s} = {v!s:28s} ({why})" for k, (v, why) in choices.items())
