"""Realce del stem de contrabajo guiado por la altura detectada.

Cadena (sobre la señal mono de análisis; en estéreo se aplica la misma máscara a
cada canal):

1. Reducción de colado percusivo (HPSS, se conserva el ataque de la pulsación).
2. Una sola pasada STFT con tres máscaras combinadas:
   * puerta espectral (ruido estacionario), que nunca atenúa los bins armónicos;
   * máscara armónica adaptativa: deja pasar k·f0 (con tolerancia creciente por
     inarmonicidad) y atenúa el resto; en tramas sin nota aplica una forma
     pasa-bajos suave para conservar los golpes/transitorios;
   * ventana de protección alrededor de cada ataque.
3. Reconstrucción de la fundamental y el 2.º armónico cuando la grabación los
   perdió (oscilador síncrono con la f0 y la envolvente de los armónicos
   superiores).
4. Ecualización (cuerpo, presencia, pasa-altos/pasa-bajos), compresión y limitación.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa
from scipy.ndimage import uniform_filter, uniform_filter1d

from .config import Settings
from .dsp import (EPS, amp2db, compressor, db2amp, highpass, istft, limiter, low_shelf,
                  lowpass, normalize_peak, peaking_eq, rms, smooth1d, stft)
from .notes import Note, note_activity
from .pitch import PitchTrack

N_FFT = 8192
HOP = 1024


@dataclass
class EnhanceInfo:
    """Información diagnóstica del realce."""
    sub_synth_rms_db: float
    voiced_fraction: float
    noise_floor_db: float


# ---------------------------------------------------------------------------
# Máscaras
# ---------------------------------------------------------------------------
def harmonic_mask(freqs: np.ndarray, f0_frames: np.ndarray, strength: float,
                  tol: float = 0.012, min_bw_bins: float = 1.5, max_harm_hz: float = 5000.0,
                  unvoiced_lp_hz: float = 250.0) -> np.ndarray:
    """Máscara (F, T) que realza los armónicos de ``f0_frames`` (NaN = sin nota)."""
    F, T = len(freqs), len(f0_frames)
    floor = float(db2amp(-30.0 * strength))          # atenuación de lo no armónico
    min_bw_hz = min_bw_bins * float(freqs[1] - freqs[0])
    mask = np.empty((F, T), dtype=np.float32)
    voiced = np.isfinite(f0_frames)
    f = freqs[:, None]
    if voiced.any():
        f0 = f0_frames[voiced][None, :]
        h = f / f0                                   # número de armónico (real)
        hn = np.maximum(1.0, np.round(h))
        dist_hz = np.abs(h - hn) * f0                # distancia al armónico más cercano
        width = min_bw_hz + tol * hn * f0 * (1.0 + 0.004 * hn ** 2)   # ensanche por inarmonicidad
        g = np.exp(-0.5 * (dist_hz / width) ** 2)
        g[freqs > max_harm_hz, :] = 0.0
        g[h < 0.5] = 0.0                             # nada por debajo de f0/2
        mask[:, voiced] = floor + (1.0 - floor) * g
    if (~voiced).any():
        # sin nota: conservar graves (golpes, ataques) y atenuar el resto
        lp = 1.0 / (1.0 + (freqs / unvoiced_lp_hz) ** 4)
        mask[:, ~voiced] = (floor + (1.0 - floor) * lp)[:, None]
    return mask


def gate_mask(mag: np.ndarray, strength: float, floor_db: float = -30.0,
              noise_percentile: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Puerta espectral suave; devuelve (ganancia (F,T), perfil de ruido (F,))."""
    if strength <= 0:
        return np.ones_like(mag), np.zeros(mag.shape[0], dtype=np.float32)
    # perfil: mínimo entre el percentil temporal por bin y la media de las tramas
    # más silenciosas (robusto frente a notas largas/pedales)
    energy = mag.sum(axis=0)
    quiet = energy <= np.percentile(energy, 12)
    prof_q = mag[:, quiet].mean(axis=1) if quiet.sum() >= 3 else np.full(mag.shape[0], np.inf)
    prof_p = np.percentile(mag, noise_percentile, axis=1)
    noise = np.minimum(prof_q, prof_p).astype(np.float32)
    noise = uniform_filter1d(noise, size=9, mode="nearest")
    thresh = noise[:, None] * (1.0 + 3.0 * strength) + EPS
    ratio = mag / thresh
    g = ratio ** 2 / (1.0 + ratio ** 2)
    floor = float(db2amp(floor_db * strength))
    g = floor + (1.0 - floor) * g
    g = uniform_filter(g, size=(3, 3), mode="nearest")
    return g.astype(np.float32), noise


def onset_protection(onsets: np.ndarray, n_frames: int, hop_s: float, pre: int = 1,
                     post: int = 2) -> np.ndarray:
    """Vector (T,) 0..1 que vale 1 alrededor de cada ataque."""
    prot = np.zeros(n_frames, dtype=np.float32)
    for t in np.asarray(onsets):
        c = int(round(t / hop_s))
        a, b = max(0, c - pre), min(n_frames, c + post + 1)
        prot[a:b] = 1.0
    return prot


# ---------------------------------------------------------------------------
# Reconstrucción de la fundamental
# ---------------------------------------------------------------------------
def _sine_calibration(n_fft: int = N_FFT) -> float:
    """Factor |STFT|_pico / amplitud para un seno con ventana Hann (librosa, sin normalizar)."""
    return n_fft / 4.0


def reconstruct_low_end(mag: np.ndarray, freqs: np.ndarray, f0_frames: np.ndarray,
                        times: np.ndarray, sr: int, n_samples: int, gain: float,
                        harm_tol: float = 0.03) -> Tuple[np.ndarray, float]:
    """Sintetiza fundamental y 2.º armónico cuando están por debajo de lo esperado.

    La amplitud objetivo de cada parcial se estima a partir de los armónicos
    presentes (2.º..5.º) con una pendiente suave (-2 dB/armónico), típica de un
    contrabajo captado de cerca. Devuelve (señal, rms_db de lo añadido).
    """
    if gain <= 0 or not np.isfinite(f0_frames).any():
        return np.zeros(n_samples, dtype=np.float32), -120.0
    bin_hz = freqs[1] - freqs[0]
    T = len(f0_frames)
    calib = _sine_calibration()

    def partial_amp(k: int) -> np.ndarray:
        """Amplitud (lineal, en unidades de forma de onda) del armónico k por trama."""
        out = np.zeros(T, dtype=np.float32)
        v = np.isfinite(f0_frames)
        fk = k * f0_frames[v]
        lo = np.clip(np.floor((fk * (1 - harm_tol)) / bin_hz).astype(int), 0, len(freqs) - 1)
        hi = np.clip(np.ceil((fk * (1 + harm_tol)) / bin_hz).astype(int) + 1, 1, len(freqs))
        cols = np.flatnonzero(v)
        vals = np.array([mag[lo[i]:hi[i], c].max() if hi[i] > lo[i] else 0.0
                         for i, c in enumerate(cols)], dtype=np.float32)
        out[v] = vals / calib
        return out

    a = {k: partial_amp(k) for k in (1, 2, 3, 4, 5)}
    # objetivo: en un contrabajo pizzicato captado de cerca la fundamental queda
    # unos 3 dB por debajo de la mediana de los armónicos 2-4 y el 2.º armónico al
    # nivel de la mediana de los armónicos 3-5 (valores conservadores).
    target1 = np.median(np.stack([a[2], a[3], a[4]]), axis=0) * float(db2amp(-3.0))
    target2 = np.median(np.stack([a[3], a[4], a[5]]), axis=0)
    add1 = np.maximum(target1 - a[1], 0.0)
    add2 = np.maximum(target2 - a[2], 0.0)
    add1 = smooth1d(add1, 3) * gain
    add2 = smooth1d(add2, 3) * gain * 0.7

    # oscilador síncrono: fase continua con f0(t) interpolada
    t_s = np.arange(n_samples) / sr
    v = np.isfinite(f0_frames)
    if v.sum() < 2:
        return np.zeros(n_samples, dtype=np.float32), -120.0
    f0_s = np.interp(t_s, times[v], f0_frames[v])
    act = np.interp(t_s, times, v.astype(np.float32))
    act = smooth1d(act, int(0.01 * sr))
    env1 = np.interp(t_s, times, add1) * act
    env2 = np.interp(t_s, times, add2) * act
    phase = 2 * np.pi * np.cumsum(f0_s) / sr
    y = env1 * np.sin(phase) + env2 * np.sin(2 * phase)
    y = y.astype(np.float32)
    return y, float(amp2db(rms(y)))


# ---------------------------------------------------------------------------
# Cadena completa
# ---------------------------------------------------------------------------
def _bleed_reduction(y: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """HPSS: atenúa la parte percusiva (bombo/caja coladas) conservando el ataque."""
    if amount <= 0:
        return y
    S = stft(y, 2048, 512)
    H, P = librosa.decompose.hpss(S, kernel_size=(17, 17), margin=(1.0, 1.0 + 2.0 * amount))
    keep = 1.0 - 0.85 * amount
    return istft(H + keep * P, 512, len(y))


def enhance_bass(stem: np.ndarray, sr: int, track: PitchTrack, notes: List[Note],
                 onsets: np.ndarray, settings: Settings) -> Tuple[np.ndarray, EnhanceInfo]:
    """Aplica la cadena de realce a ``stem`` (mono (N,) o estéreo (2, N))."""
    stem = np.asarray(stem, dtype=np.float32)
    single = stem.ndim == 1
    chans = stem[None, :] if single else stem
    n = chans.shape[-1]
    mono = chans.mean(axis=0)

    # 1. colado percusivo
    if settings.bleed_reduction > 0:
        chans = np.stack([_bleed_reduction(c, sr, settings.bleed_reduction) for c in chans])
        mono = chans.mean(axis=0)

    # 2. máscaras en una pasada STFT
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    S_mono = stft(mono, N_FFT, HOP)
    mag = np.abs(S_mono)
    T = mag.shape[1]
    frame_times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=HOP)
    f0_frames = track.f0_at(frame_times)

    hmask = harmonic_mask(freqs, f0_frames, settings.harmonic_strength)
    gmask, noise_prof = gate_mask(mag, settings.denoise)
    # proteger los bins armónicos de la puerta
    gmask = np.where(hmask > 0.5, np.maximum(gmask, hmask), gmask)
    prot = onset_protection(onsets, T, HOP / sr)
    mask = hmask * gmask
    lp_prot = (1.0 / (1.0 + (freqs / 3000.0) ** 4))[:, None]
    mask = np.maximum(mask, prot[None, :] * lp_prot)
    mask = uniform_filter(mask, size=(1, 2), mode="nearest").astype(np.float32)

    out = np.empty_like(chans)
    for c in range(chans.shape[0]):
        S = S_mono if chans.shape[0] == 1 else stft(chans[c], N_FFT, HOP)
        out[c] = istft(S * mask, HOP, n)

    # 3. fundamental perdida
    sub, sub_db = reconstruct_low_end(mag, freqs, f0_frames, frame_times, sr, n,
                                      settings.sub_gain)
    if settings.sub_gain > 0:
        out = out + sub[None, :]

    # 4. EQ, dinámica
    out = highpass(out, sr, settings.highpass_hz, order=2)
    out = lowpass(out, sr, settings.lowpass_hz, order=4)
    out = low_shelf(out, sr, 90.0, settings.body_db)
    out = peaking_eq(out, sr, 800.0, settings.presence_db, q=1.2)
    if settings.compress > 0:
        thr = amp2db(rms(out)) + 8.0 - 6.0 * settings.compress
        out = compressor(out, sr, threshold_db=float(thr), ratio=1.5 + 2.5 * settings.compress,
                         attack_ms=12.0, release_ms=180.0)
    out = normalize_peak(out, settings.target_peak_db)
    out = limiter(out, sr, ceiling_db=settings.target_peak_db)

    info = EnhanceInfo(sub_synth_rms_db=sub_db, voiced_fraction=track.voiced_fraction(),
                       noise_floor_db=float(amp2db(noise_prof.mean() / _sine_calibration() + EPS)))
    return (out[0] if single else out), info
