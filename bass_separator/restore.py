"""Restauración previa para transferencias antiguas (78 rpm, cinta, acetato)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation, uniform_filter1d

from .config import Settings
from .dsp import highpass, notch, spectral_gate


@dataclass
class RestoreInfo:
    clicks_removed: int = 0
    hum_hz: float = 0.0
    hum_notched: bool = False
    pre_denoise: float = 0.0


def declick(y: np.ndarray, sr: int, threshold: float = 7.0, max_width_ms: float = 1.5,
            context: int = 12) -> Tuple[np.ndarray, int]:
    """Detecta impulsos (clics/crepitar) y los repara por interpolación cúbica."""
    y = np.asarray(y, dtype=np.float32)
    single = y.ndim == 1
    chans = y[None, :] if single else y
    out = chans.copy()
    max_w = max(2, int(max_width_ms * sr / 1000))
    n_total = 0
    for c in range(chans.shape[0]):
        x = chans[c].astype(np.float64)
        sos = signal.butter(2, min(2500.0, sr / 2 * 0.8), btype="highpass", fs=sr, output="sos")
        r = signal.sosfilt(sos, x)
        # umbral adaptativo: MAD global y envolvente local (20 ms)
        mad = np.median(np.abs(r)) * 1.4826 + 1e-9
        local = np.sqrt(uniform_filter1d(r ** 2, size=int(0.02 * sr) + 1, mode="nearest"))
        thr = threshold * np.maximum(mad, local)
        mask = np.abs(r) > thr
        if not mask.any():
            continue
        mask = binary_dilation(mask, iterations=3)
        # localizar segmentos
        d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        n = len(x)
        for a, b in zip(starts, ends):
            if b - a > max_w:
                continue
            lo, hi = max(0, a - context), min(n, b + context)
            if lo >= a or hi <= b:
                continue
            idx = np.concatenate((np.arange(lo, a), np.arange(b, hi)))
            cs = CubicSpline(idx, x[idx])
            x[a:b] = cs(np.arange(a, b))
            n_total += 1
        out[c] = x.astype(np.float32)
    return (out[0] if single else out), n_total


def detect_hum(y: np.ndarray, sr: int, candidates=(50.0, 60.0), min_prominence_db: float = 15.0,
               seg_s: float = 10.0) -> float:
    """Devuelve 50 o 60 si hay un zumbido de red persistente, 0 en caso contrario."""
    mono = y if y.ndim == 1 else y.mean(axis=0)
    n_seg = max(1, int(len(mono) / (seg_s * sr)))
    votes = {c: 0 for c in candidates}
    for i in range(n_seg):
        seg = mono[int(i * seg_s * sr):int((i + 1) * seg_s * sr)]
        if len(seg) < sr * 2:
            continue
        f, p = signal.welch(seg, fs=sr, nperseg=1 << 16)
        pdb = 10 * np.log10(p + 1e-20)
        for c in candidates:
            band = (f > c - 0.7) & (f < c + 0.7)
            around = ((f > c - 8) & (f < c - 2)) | ((f > c + 2) & (f < c + 8))
            if band.any() and around.any() and pdb[band].max() - np.median(pdb[around]) > min_prominence_db:
                votes[c] += 1
    best = max(votes, key=votes.get)
    return float(best) if votes[best] >= 0.8 * n_seg and votes[best] > 0 else 0.0


def remove_hum(y: np.ndarray, sr: int, base: float, harmonics: int = 3, q: float = 35.0) -> np.ndarray:
    out = y
    for k in range(1, harmonics + 1):
        out = notch(out, sr, base * k, q=q)
    return out


def pre_restore(y: np.ndarray, sr: int, settings: Settings) -> Tuple[np.ndarray, RestoreInfo]:
    """Cadena previa: DC/subsónicos, de-click, zumbido y siseo (suave)."""
    info = RestoreInfo()
    out = np.asarray(y, dtype=np.float32)
    out = highpass(out, sr, 20.0, order=2)
    if settings.declick:
        out, info.clicks_removed = declick(out, sr)
    if settings.hum_notch:
        hum = detect_hum(out, sr)
        info.hum_hz = hum
        if hum > 0:
            out = remove_hum(out, sr, hum)
            info.hum_notched = True
    if settings.pre_denoise > 0:
        strength = 0.6 * float(np.clip(settings.pre_denoise, 0.0, 1.0))
        out = spectral_gate(out, sr, strength, floor_db=-18.0)
        info.pre_denoise = strength
    return out, info
