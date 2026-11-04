"""Seguimiento de la frecuencia fundamental (f0) de una línea de contrabajo.

Dos métodos:

* ``salience``: suma armónica ponderada sobre una rejilla logarítmica de
  candidatos + decodificación de Viterbi. Diseñado para grabaciones antiguas
  donde la fundamental (40-80 Hz) suele estar atenuada o ausente: la altura se
  infiere a partir de los armónicos superiores y se penalizan los errores de
  octava comprobando la ausencia de sub-armónicos.
* ``pyin``: implementación probabilística YIN de librosa (referencia).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa
from scipy.ndimage import maximum_filter1d, uniform_filter1d, median_filter

from .audio_io import resample, to_mono
from .dsp import amp2db, block_stft_mag, highpass, hz_to_midi, frame_rms_db

ANALYSIS_SR = 11025


@dataclass
class PitchTrack:
    """Trayectoria de f0 por tramas (``f0`` es NaN donde no hay nota)."""

    times: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    rms_db: np.ndarray
    hop_s: float

    @property
    def midi(self) -> np.ndarray:
        out = np.full_like(self.f0, np.nan)
        v = self.voiced
        out[v] = hz_to_midi(self.f0[v])
        return out

    def f0_at(self, t: np.ndarray) -> np.ndarray:
        """f0 interpolada en los instantes ``t`` (NaN donde la trama más cercana no es sonora)."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.round(t / self.hop_s).astype(int), 0, len(self.f0) - 1)
        out = np.full(t.shape, np.nan)
        v = self.voiced[idx]
        if v.any():
            vi = np.flatnonzero(self.voiced)
            interp = np.interp(t[v], self.times[vi], self.f0[vi])
            out[v] = interp
        return out

    def voiced_fraction(self) -> float:
        return float(np.mean(self.voiced)) if len(self.voiced) else 0.0


def _prepare(y: np.ndarray, sr: int, hop_s: float):
    ya = to_mono(np.asarray(y, dtype=np.float32))
    if sr != ANALYSIS_SR:
        ya = resample(ya, sr, ANALYSIS_SR)
    ya = highpass(ya, ANALYSIS_SR, 25.0, order=2)
    hop = max(1, int(round(hop_s * ANALYSIS_SR)))
    return ya, hop


def _postprocess(f0: np.ndarray, voiced: np.ndarray, min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Elimina islas sonoras muy cortas y suaviza f0 con mediana dentro de cada nota."""
    voiced = voiced.copy()
    f0 = f0.copy()
    n = len(voiced)
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if j - i < min_frames:
                voiced[i:j] = False
            elif j - i >= 3:
                f0[i:j] = median_filter(f0[i:j], size=3, mode="nearest")
            i = j
        else:
            i += 1
    f0[~voiced] = np.nan
    return f0, voiced


# ---------------------------------------------------------------------------
# Método "salience"
# ---------------------------------------------------------------------------
def _peakiness(mag: np.ndarray, floor_bins: int = 75, margin_db: float = 3.0,
               range_db: float = 18.0) -> np.ndarray:
    """Mapa 0..1 de "prominencia de pico" por bin y trama (blanqueo espectral)."""
    L = amp2db(mag).astype(np.float32)
    floor = uniform_filter1d(L, size=floor_bins, axis=0, mode="nearest")
    P = np.clip((L - floor - margin_db) / range_db, 0.0, 1.0)
    # solo máximos locales en frecuencia (con tolerancia de +-1 bin)
    local_max = maximum_filter1d(L, size=3, axis=0, mode="nearest")
    P[L < local_max - 0.5] = 0.0
    return P


def _salience_matrix(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float,
                     cents_step: float = 20.0, n_harm: int = 10, harm_decay: float = 0.8,
                     sub_penalty: float = 0.8, margin_db: float = 3.0):
    """Matriz de salience (candidatos x tramas) y datos auxiliares."""
    ya, hop = _prepare(y, sr, hop_s)
    sr_a = ANALYSIS_SR
    win, n_fft = 2048, 4096
    bin_hz = sr_a / n_fft
    f_top = min(fmax * (n_harm + 0.5), sr_a / 2 - 50)
    max_bin = int(f_top / bin_hz) + 3
    mag = block_stft_mag(ya, n_fft, hop, win, max_bin)           # (F, T)
    n_frames = mag.shape[1]
    P = _peakiness(mag, margin_db=margin_db)
    # tolerancia creciente con el número de armónico (inarmonicidad de la cuerda)
    Pd = {1: maximum_filter1d(P, size=5, axis=0, mode="nearest"),
          2: maximum_filter1d(P, size=9, axis=0, mode="nearest"),
          3: maximum_filter1d(P, size=15, axis=0, mode="nearest")}

    n_cand = int(np.ceil(1200.0 * np.log2(fmax / fmin) / cents_step)) + 1
    cand = fmin * 2.0 ** (np.arange(n_cand) * cents_step / 1200.0)
    w = np.arange(1, n_harm + 1) ** (-harm_decay)
    w_sum = w.sum()

    def lookup(freqs: np.ndarray, _level: int = 0) -> np.ndarray:
        # tolerancia proporcional a la frecuencia (~1 %) + 1 bin
        idx = np.clip(np.round(freqs / bin_hz).astype(int), 0, max_bin - 1)
        f = np.asarray(freqs)[:, None]
        out = np.where(f < 700.0, Pd[1][idx], np.where(f < 1500.0, Pd[2][idx], Pd[3][idx]))
        return out

    sal = np.zeros((n_cand, n_frames), dtype=np.float32)
    for k in range(1, n_harm + 1):
        sal += (w[k - 1] / w_sum) * lookup(k * cand)
    # Penalizaciones de errores de octava (cand = 2 f0 real) y de duodécima
    # (cand = 3 f0 real): si hay picos simultáneos en 3/2 y 5/2 del candidato
    # (o en 2/3 y 4/3) el candidato es un múltiplo de la verdadera f0. Se usa el
    # mínimo de dos posiciones para que un pico aislado (cuerpo, bombo) no dispare
    # la penalización, y no se usa f0/2 porque la fundamental puede faltar.
    pen2 = np.minimum(lookup(1.5 * cand), lookup(2.5 * cand))
    pen2 *= (cand / 2 >= fmin * 0.97)[:, None]
    pen3 = np.minimum(lookup(cand * 2 / 3), lookup(cand * 4 / 3))
    pen3 *= (cand / 3 >= fmin * 0.97)[:, None]
    # Penalización de sub-octava (cand = f0 real / 2): los armónicos pares del
    # candidato están presentes y los impares (3, 5, 7) ausentes.
    even = (lookup(2 * cand) + lookup(4 * cand) + lookup(6 * cand)) / 3
    odd = (lookup(3 * cand) + lookup(5 * cand) + lookup(7 * cand)) / 3
    pen_sub = np.clip(even - odd - 0.2, 0.0, 1.0)
    pen = np.maximum(np.maximum(pen2, pen3), pen_sub)
    sal = np.clip(sal - sub_penalty * pen, 0.0, 1.0)

    rdb = frame_rms_db(ya, win, hop)[:n_frames]
    if len(rdb) < n_frames:
        rdb = np.pad(rdb, (0, n_frames - len(rdb)), mode="edge")
    return cand, sal, rdb, hop


def _salience_track(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float,
                    cents_step: float = 20.0, n_harm: int = 10, harm_decay: float = 0.8,
                    sub_penalty: float = 0.8, voicing_threshold: float = 0.3,
                    emission_sharpness: float = 6.0, jump_tau: float = 2.0,
                    p_stay: float = 0.85, energy_gate_db: float = 50.0,
                    margin_db: float = 3.0) -> PitchTrack:
    cand, sal, rdb, hop = _salience_matrix(y, sr, fmin, fmax, hop_s, cents_step, n_harm,
                                           harm_decay, sub_penalty, margin_db)
    sr_a = ANALYSIS_SR
    n_cand, n_frames = sal.shape
    ref = np.percentile(rdb, 98)
    quiet = rdb < ref - energy_gate_db

    # emisiones log-lineales: p(obs | f0) ∝ exp(lambda * salience)
    emis = np.empty((n_cand + 1, n_frames), dtype=np.float64)
    emis[:n_cand] = np.exp(emission_sharpness * sal.astype(np.float64))
    emis[n_cand] = np.exp(emission_sharpness * voicing_threshold)
    emis[:n_cand, quiet] *= 0.05
    emis /= emis.sum(axis=0, keepdims=True)

    # transiciones: permanencia alta, saltos con coste creciente en semitonos
    st = np.arange(n_cand) * cents_step / 100.0
    d = np.abs(st[:, None] - st[None, :])
    off = np.exp(-d / jump_tau) + 1e-4
    np.fill_diagonal(off, 0.0)
    off /= off.sum(axis=1, keepdims=True)
    trans = (1.0 - p_stay) * off + p_stay * np.eye(n_cand)
    p_vu, p_uv = 0.03, 0.03
    T = np.zeros((n_cand + 1, n_cand + 1))
    T[:n_cand, :n_cand] = trans * (1 - p_vu)
    T[:n_cand, n_cand] = p_vu
    T[n_cand, :n_cand] = p_uv / n_cand
    T[n_cand, n_cand] = 1 - p_uv
    p_init = np.full(n_cand + 1, 0.5 / n_cand)
    p_init[n_cand] = 0.5
    states = librosa.sequence.viterbi(emis, T, p_init=p_init)

    voiced = states < n_cand
    f0 = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    vi = np.flatnonzero(voiced)
    if len(vi):
        s = states[vi]
        # refinado parabólico en la rejilla logarítmica
        sm = np.clip(s - 1, 0, n_cand - 1)
        sp = np.clip(s + 1, 0, n_cand - 1)
        a, b, c = sal[sm, vi], sal[s, vi], sal[sp, vi]
        denom = a - 2 * b + c
        safe = np.where(np.abs(denom) > 1e-6, denom, 1.0)
        delta = np.where(np.abs(denom) > 1e-6, 0.5 * (a - c) / safe, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        f0[vi] = fmin * 2.0 ** ((s + delta) * cents_step / 1200.0)
        conf[vi] = b
    f0, voiced = _postprocess(f0, voiced)
    conf[~voiced] = 0.0
    times = np.arange(n_frames) * hop / sr_a
    return PitchTrack(times=times, f0=f0, voiced=voiced, confidence=conf, rms_db=rdb,
                      hop_s=hop / sr_a)


# ---------------------------------------------------------------------------
# Método "pyin"
# ---------------------------------------------------------------------------
def _pyin_track(y: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float) -> PitchTrack:
    ya, hop = _prepare(y, sr, hop_s)
    sr_a = ANALYSIS_SR
    frame = 2048
    f0, vflag, vprob = librosa.pyin(ya, fmin=fmin, fmax=fmax, sr=sr_a, frame_length=frame,
                                    hop_length=hop, fill_na=np.nan)
    n_frames = len(f0)
    rdb = frame_rms_db(ya, frame, hop)[:n_frames]
    if len(rdb) < n_frames:
        rdb = np.pad(rdb, (0, n_frames - len(rdb)), mode="edge")
    voiced = np.asarray(vflag, dtype=bool) & np.isfinite(f0)
    f0 = np.where(voiced, f0, np.nan)
    f0, voiced = _postprocess(f0, voiced)
    conf = np.where(voiced, vprob, 0.0)
    times = np.arange(n_frames) * hop / sr_a
    return PitchTrack(times=times, f0=f0, voiced=voiced, confidence=conf, rms_db=rdb,
                      hop_s=hop / sr_a)


def track_pitch(y: np.ndarray, sr: int, fmin: float = 38.0, fmax: float = 330.0,
                method: str = "salience", hop_s: float = 0.01, **kwargs) -> PitchTrack:
    """Estima la trayectoria de f0 del contrabajo en ``y`` (mono o multicanal)."""
    if method == "salience":
        return _salience_track(y, sr, fmin, fmax, hop_s, **kwargs)
    if method == "pyin":
        return _pyin_track(y, sr, fmin, fmax, hop_s)
    raise ValueError(f"Método de pitch desconocido: {method!r}")
