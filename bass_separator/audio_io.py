"""Carga, guardado y remuestreo de audio (solo libsndfile + scipy)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Lee un archivo de audio y devuelve ``(y, sr)`` con ``y`` de forma (canales, muestras) float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = np.ascontiguousarray(data.T)
    return y, int(sr)


def save_audio(path: str | Path, y: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """Guarda ``y`` (canales, muestras) o (muestras,) como WAV."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if subtype.startswith("PCM"):
        y = np.clip(y, -1.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.T, sr, subtype=subtype)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Remuestreo polifásico de alta calidad (funciona por canal, último eje = tiempo)."""
    if sr_in == sr_out:
        return y
    frac = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    out = resample_poly(y, frac.numerator, frac.denominator, axis=-1)
    return np.asarray(out, dtype=np.float32)


def to_mono(y: np.ndarray) -> np.ndarray:
    """Mezcla (canales, muestras) a (muestras,)."""
    if y.ndim == 1:
        return y
    return y.mean(axis=0).astype(np.float32)


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0)
    if y.shape[0] > 2:
        return y[:2]
    return y


def is_effectively_mono(y: np.ndarray, tol_db: float = -40.0) -> bool:
    """True si los canales son (casi) idénticos: típico de transferencias de 78 rpm."""
    if y.ndim == 1 or y.shape[0] == 1:
        return True
    l, r = y[0], y[1]
    diff = np.mean((l - r) ** 2)
    total = np.mean((l + r) ** 2) / 4 + 1e-12
    return 10 * np.log10(diff / total + 1e-12) < tol_db


def match_length(y: np.ndarray, n: int) -> np.ndarray:
    """Recorta o rellena con ceros el último eje hasta ``n`` muestras."""
    cur = y.shape[-1]
    if cur == n:
        return y
    if cur > n:
        return y[..., :n]
    pad = [(0, 0)] * (y.ndim - 1) + [(0, n - cur)]
    return np.pad(y, pad)


def speed_correct(y: np.ndarray, sr: int, cents: float) -> np.ndarray:
    """Corrige la velocidad de reproducción (afinación y tempo a la vez).

    ``cents`` > 0 significa que la grabación suena aguda (va rápida): se
    alarga la señal por 2^(cents/1200) para bajarla a la afinación de referencia.
    """
    if abs(cents) < 1e-6:
        return y
    factor = 2.0 ** (cents / 1200.0)
    return resample(y, sr, int(round(sr * factor)))
