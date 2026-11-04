"""Modelo físico sencillo de contrabajo pizzicato (Karplus-Strong extendido + cuerpo).

Se usa para la capa de resíntesis: cada nota detectada se vuelve a "tocar" con un
sonido limpio de contrabajo, respetando el instante, la duración, la afinación
real y la dinámica del intérprete.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy import signal

from .dsp import highpass, lowpass, normalize_peak


class DoubleBassSynth:
    """Sintetizador de contrabajo pulsado.

    Parameters
    ----------
    sr : frecuencia de muestreo de salida.
    brightness : 0..1, brillo del ataque (filtro del excitador).
    body : cantidad de resonancia de cuerpo (0 = solo cuerda).
    """

    # modos de resonancia aproximados de un contrabajo (Hz, Q, ganancia)
    BODY_MODES: Tuple[Tuple[float, float, float], ...] = (
        (62.0, 6.0, 0.9), (98.0, 8.0, 0.7), (165.0, 10.0, 0.5),
        (230.0, 8.0, 0.35), (400.0, 5.0, 0.25), (640.0, 5.0, 0.2),
    )

    def __init__(self, sr: int = 44100, brightness: float = 0.5, body: float = 1.0, seed: int = 0):
        self.sr = int(sr)
        self.brightness = float(np.clip(brightness, 0.0, 1.0))
        self.body = float(body)
        self.rng = np.random.default_rng(seed)
        self._body_sos = [signal.iirpeak(f, q, fs=self.sr) for f, q, _ in self.BODY_MODES]

    # -- cuerda -----------------------------------------------------------------
    def pluck(self, f0: float, duration: float, velocity: float = 0.8,
              release: float = 0.05) -> np.ndarray:
        """Una nota: ``duration`` segundos sonando y luego una caída de ``release`` s."""
        sr = self.sr
        f0 = float(np.clip(f0, 20.0, 1000.0))
        velocity = float(np.clip(velocity, 0.05, 1.0))
        stretch = 0.5 - 0.35 * self.brightness * velocity          # 0.5 = máxima pérdida de agudos
        N = max(2, int(round(sr / f0 - stretch)))
        n_total = int((duration + release) * sr) + N
        # tiempo de decaimiento: más largo en graves
        t60 = 1.6 * (41.2 / f0) ** 0.35
        g = 10.0 ** (-3.0 * N / (sr * t60))
        # excitación: ráfaga de ruido filtrada + posición de pulsación
        exc = self.rng.uniform(-1.0, 1.0, N).astype(np.float64)
        cutoff = 300.0 + 2500.0 * self.brightness * velocity
        b, a = signal.butter(1, min(cutoff, sr / 2 * 0.9), fs=sr)
        exc = signal.lfilter(b, a, exc)
        pos = max(1, int(0.13 * N))
        exc[pos:] -= 0.9 * exc[:-pos]
        exc -= exc.mean()
        x = np.zeros(n_total)
        x[:N] = exc
        # bucle Karplus-Strong estirado: y[n] = x[n] + g*((1-S) y[n-N] + S y[n-N-1])
        den = np.zeros(N + 2)
        den[0] = 1.0
        den[N] = -g * (1.0 - stretch)
        den[N + 1] = -g * stretch
        y = signal.lfilter([1.0], den, x)
        # amortiguación al final de la nota (el siguiente dedo apaga la cuerda)
        n_on = int(duration * sr)
        env = np.ones(n_total)
        n_rel = n_total - n_on
        if n_rel > 0:
            env[n_on:] = np.exp(-np.arange(n_rel) / (0.25 * release * sr + 1))
        y *= env
        # pequeño golpe del dedo al atacar
        thump_n = int(0.006 * sr)
        if thump_n > 0:
            thump = self.rng.normal(0, 1, thump_n)
            bt, at = signal.butter(2, 160.0, fs=sr)
            thump = signal.lfilter(bt, at, thump) * np.linspace(1, 0, thump_n) ** 2
            y[:thump_n] += 0.4 * velocity * thump / (np.max(np.abs(thump)) + 1e-9) * np.max(np.abs(y[:N * 2] if N * 2 < n_total else y))
        y = normalize_peak(y.astype(np.float32), 0.0) * velocity ** 1.4
        return y

    # -- cuerpo -----------------------------------------------------------------
    def body_filter(self, y: np.ndarray) -> np.ndarray:
        if self.body <= 0:
            return y
        out = y.astype(np.float64).copy()
        for (b, a), (_, _, gain) in zip(self._body_sos, self.BODY_MODES):
            out += self.body * gain * signal.lfilter(b, a, y)
        return out.astype(np.float32)

    # -- render ----------------------------------------------------------------
    def render(self, notes: Iterable, duration: float, max_sustain: float = 1.2) -> np.ndarray:
        """Mezcla todas las notas en un buffer mono de ``duration`` segundos."""
        sr = self.sr
        n = int(np.ceil(duration * sr)) + 1
        out = np.zeros(n, dtype=np.float32)
        notes = list(notes)
        for i, nt in enumerate(notes):
            dur = max(0.03, min(nt.duration, max_sustain))
            vel = float(np.clip(getattr(nt, "velocity", 100) / 127.0, 0.1, 1.0))
            tone = self.pluck(nt.f0, dur, vel)
            a = int(nt.start * sr)
            b = min(n, a + len(tone))
            if b > a:
                out[a:b] += tone[:b - a]
        if not notes:
            return out
        out = self.body_filter(out)
        out = highpass(out, sr, 30.0, order=2)
        out = lowpass(out, sr, 4000.0, order=2)
        return normalize_peak(out, -3.0)
