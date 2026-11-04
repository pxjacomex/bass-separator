"""Utilidades DSP: filtros, puerta espectral, dinámica y STFT por bloques.

Solo depende de numpy/scipy/librosa (numba se usa si está disponible, que es
lo habitual porque librosa lo requiere).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter, uniform_filter1d, minimum_filter1d

EPS = 1e-10


# ---------------------------------------------------------------------------
# Conversión de unidades
# ---------------------------------------------------------------------------
def db2amp(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def amp2db(a: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(a, dtype=np.float64), EPS))


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + EPS))


def rms_db(y: np.ndarray) -> float:
    return float(amp2db(rms(y)))


def hz_to_midi(f: np.ndarray | float) -> np.ndarray | float:
    return 69.0 + 12.0 * np.log2(np.maximum(np.asarray(f, dtype=np.float64), EPS) / 440.0)


def midi_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def highpass(y: np.ndarray, sr: int, fc: float, order: int = 2) -> np.ndarray:
    if fc <= 0:
        return y
    sos = signal.butter(order, fc, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y, axis=-1).astype(np.float32)


def lowpass(y: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    if fc <= 0 or fc >= sr / 2:
        return y
    sos = signal.butter(order, fc, btype="lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y, axis=-1).astype(np.float32)


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float, order: int = 2) -> np.ndarray:
    hi = min(hi, sr / 2 * 0.99)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y, axis=-1).astype(np.float32)


def _rbj_peaking(sr: int, fc: float, gain_db: float, q: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b0, b1, b2 = 1 + alpha * a, -2 * cw, 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * cw, 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _rbj_shelf(sr: int, fc: float, gain_db: float, low: bool, slope: float = 1.0) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2 * math.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    sq = 2 * math.sqrt(a) * alpha
    if low:
        b0 = a * ((a + 1) - (a - 1) * cw + sq)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - sq)
        a0 = (a + 1) + (a - 1) * cw + sq
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - sq
    else:
        b0 = a * ((a + 1) + (a - 1) * cw + sq)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - sq)
        a0 = (a + 1) - (a - 1) * cw + sq
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - sq
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def peaking_eq(y: np.ndarray, sr: int, fc: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return y
    return signal.sosfilt(_rbj_peaking(sr, fc, gain_db, q), y, axis=-1).astype(np.float32)


def low_shelf(y: np.ndarray, sr: int, fc: float, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return y
    return signal.sosfilt(_rbj_shelf(sr, fc, gain_db, low=True), y, axis=-1).astype(np.float32)


def high_shelf(y: np.ndarray, sr: int, fc: float, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return y
    return signal.sosfilt(_rbj_shelf(sr, fc, gain_db, low=False), y, axis=-1).astype(np.float32)


def notch(y: np.ndarray, sr: int, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(f0, q, fs=sr)
    return signal.filtfilt(b, a, y, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def stft(y: np.ndarray, n_fft: int, hop: int, win_length: Optional[int] = None) -> np.ndarray:
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, window="hann", center=True)


def istft(S: np.ndarray, hop: int, length: int, win_length: Optional[int] = None) -> np.ndarray:
    n_fft = 2 * (S.shape[0] - 1)
    return librosa.istft(S, hop_length=hop, win_length=win_length, window="hann", center=True,
                         length=length).astype(np.float32)


def block_stft_mag(y: np.ndarray, n_fft: int, hop: int, win_length: int, max_bin: int,
                   block_frames: int = 4096) -> np.ndarray:
    """|STFT| limitada a ``max_bin`` bins, calculada por bloques para ahorrar memoria.

    Equivale a ``abs(librosa.stft(y, ...))[:max_bin]`` (center=True).
    """
    n = len(y)
    n_frames = 1 + n // hop
    out = np.empty((max_bin, n_frames), dtype=np.float32)
    pad = n_fft // 2
    ypad = np.pad(y, (pad, pad), mode="constant")
    win = signal.get_window("hann", win_length, fftbins=True).astype(np.float32)
    # ventana centrada dentro de n_fft (igual que librosa)
    lpad = (n_fft - win_length) // 2
    for start in range(0, n_frames, block_frames):
        stop = min(n_frames, start + block_frames)
        idx = np.arange(start, stop) * hop
        frames = np.lib.stride_tricks.sliding_window_view(ypad, n_fft)[idx]  # (B, n_fft)
        seg = frames[:, lpad:lpad + win_length] * win
        spec = np.fft.rfft(seg, n=n_fft, axis=-1)[:, :max_bin]
        out[:, start:stop] = np.abs(spec).T
    return out


# ---------------------------------------------------------------------------
# Reducción de ruido por puerta espectral
# ---------------------------------------------------------------------------
def spectral_gate(y: np.ndarray, sr: int, strength: float = 0.7, n_fft: int = 2048,
                  hop: int = 512, floor_db: float = -30.0, noise_percentile: float = 20.0,
                  smooth_t: int = 3, smooth_f: int = 3,
                  noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
    """Reduce ruido estacionario (siseo, superficie de disco) con una máscara suave.

    ``strength`` en 0..1 controla cuánto se sube el umbral por encima del perfil
    de ruido y el suelo de atenuación. 0 devuelve la señal intacta.
    """
    if strength <= 0:
        return y
    y = np.asarray(y, dtype=np.float32)
    single = y.ndim == 1
    ych = y[None, :] if single else y
    out = np.empty_like(ych)
    for c in range(ych.shape[0]):
        S = stft(ych[c], n_fft, hop)
        mag = np.abs(S)
        if noise_profile is None:
            noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
        else:
            noise = noise_profile.reshape(-1, 1)
        thresh = noise * (1.0 + 4.0 * strength) + EPS
        ratio = mag / thresh
        gain = ratio ** 2 / (1.0 + ratio ** 2)
        floor = float(db2amp(floor_db * strength))
        gain = floor + (1.0 - floor) * gain
        if smooth_t > 1 or smooth_f > 1:
            gain = uniform_filter(gain, size=(smooth_f, smooth_t), mode="nearest")
        out[c] = istft(S * gain, hop, ych.shape[-1])
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Seguidores de envolvente y dinámica
# ---------------------------------------------------------------------------
try:  # numba viene con librosa; si no está, usamos un bucle en Python puro
    from numba import njit

    @njit(cache=True)
    def _follow(x: np.ndarray, a_att: float, a_rel: float) -> np.ndarray:  # pragma: no cover
        out = np.empty_like(x)
        e = 0.0
        for i in range(x.shape[0]):
            v = x[i]
            if v > e:
                e = a_att * e + (1.0 - a_att) * v
            else:
                e = a_rel * e + (1.0 - a_rel) * v
            out[i] = e
        return out
except Exception:  # pragma: no cover
    def _follow(x: np.ndarray, a_att: float, a_rel: float) -> np.ndarray:
        out = np.empty_like(x)
        e = 0.0
        for i in range(x.shape[0]):
            v = x[i]
            e = a_att * e + (1.0 - a_att) * v if v > e else a_rel * e + (1.0 - a_rel) * v
            out[i] = e
        return out


def envelope_follower(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Envolvente de |x| con ataque/liberación exponenciales (x de forma (N,))."""
    a_att = math.exp(-1.0 / (sr * max(attack_ms, 0.01) / 1000.0))
    a_rel = math.exp(-1.0 / (sr * max(release_ms, 0.01) / 1000.0))
    return _follow(np.abs(x).astype(np.float64), a_att, a_rel)


def compressor(y: np.ndarray, sr: int, threshold_db: float, ratio: float = 3.0,
               attack_ms: float = 10.0, release_ms: float = 150.0, knee_db: float = 6.0,
               makeup: str | float = "auto") -> np.ndarray:
    """Compresor feed-forward con rodilla suave y detección enlazada entre canales."""
    y = np.asarray(y, dtype=np.float32)
    single = y.ndim == 1
    ych = y[None, :] if single else y
    det = np.max(np.abs(ych), axis=0)
    env = envelope_follower(det, sr, attack_ms, release_ms)
    lvl = amp2db(env)
    over = lvl - threshold_db
    gain_db = np.zeros_like(lvl)
    if knee_db > 0:
        in_knee = np.abs(over) <= knee_db / 2
        gain_db[in_knee] = -(1 - 1 / ratio) * (over[in_knee] + knee_db / 2) ** 2 / (2 * knee_db)
    above = over > knee_db / 2
    gain_db[above] = -(1 - 1 / ratio) * over[above]
    g = db2amp(gain_db).astype(np.float32)
    out = ych * g[None, :]
    if makeup == "auto":
        mk = rms(ych) / (rms(out) + EPS)
    else:
        mk = float(db2amp(float(makeup)))
    out = out * mk
    return out[0] if single else out


def limiter(y: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 80.0) -> np.ndarray:
    """Limitador de picos con anticipación (lookahead) y liberación suave."""
    y = np.asarray(y, dtype=np.float32)
    single = y.ndim == 1
    ych = y[None, :] if single else y
    ceiling = float(db2amp(ceiling_db))
    peak = np.max(np.abs(ych), axis=0)
    req = np.minimum(1.0, ceiling / (peak + EPS))
    la = max(1, int(sr * lookahead_ms / 1000))
    req = minimum_filter1d(req, size=2 * la + 1, mode="nearest")
    red = 1.0 - req
    red = _follow(red.astype(np.float64), math.exp(-1.0 / (sr * lookahead_ms / 2000.0 + 1)),
                  math.exp(-1.0 / (sr * release_ms / 1000.0)))
    # la anticipación garantiza que la reducción ya está aplicada al llegar el pico
    red = np.maximum(red, 1.0 - req)
    out = ych * (1.0 - red)[None, :].astype(np.float32)
    out = np.clip(out, -ceiling, ceiling)
    return out[0] if single else out


def normalize_peak(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= EPS:
        return y
    return (y * (float(db2amp(peak_db)) / peak)).astype(np.float32)


def match_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
    cur = rms(y)
    if cur <= EPS:
        return y
    return (y * (target_rms / cur)).astype(np.float32)


def fade_edges(y: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    n = int(sr * ms / 1000)
    if n <= 0 or y.shape[-1] < 2 * n:
        return y
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    y = y.copy()
    y[..., :n] *= ramp
    y[..., -n:] *= ramp[::-1]
    return y


def smooth1d(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return uniform_filter1d(x, size=n, mode="nearest")


def frame_rms_db(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS por trama (dB) alineado con STFT center=True."""
    r = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    return amp2db(r)
