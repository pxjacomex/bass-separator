"""Segmentación de la trayectoria de f0 en notas (línea de walking bass)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import librosa

from .audio_io import resample, to_mono
from .dsp import bandpass, hz_to_midi
from .pitch import PitchTrack, ANALYSIS_SR


@dataclass
class Note:
    start: float
    end: float
    midi: int
    f0: float          # Hz (mediana real, conserva la afinación del intérprete)
    velocity: int
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return librosa.midi_to_note(self.midi)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        d["duration"] = round(self.duration, 4)
        return d


def detect_onsets(y: np.ndarray, sr: int, hop_s: float = 0.01, lo: float = 40.0,
                  hi: float = 1200.0, min_gap_s: float = 0.06) -> np.ndarray:
    """Instantes de ataque (segundos) de la banda del contrabajo."""
    ya = to_mono(np.asarray(y, dtype=np.float32))
    if sr != ANALYSIS_SR:
        ya = resample(ya, sr, ANALYSIS_SR)
    sr_a = ANALYSIS_SR
    ya = bandpass(ya, sr_a, lo, min(hi, sr_a / 2 - 100), order=2)
    hop = max(1, int(round(hop_s * sr_a)))
    env = librosa.onset.onset_strength(y=ya, sr=sr_a, hop_length=hop, n_fft=1024,
                                       aggregate=np.median, lag=2)
    if env.size == 0 or np.max(env) <= 0:
        return np.zeros(0)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr_a, hop_length=hop,
                                        backtrack=True, units="frames",
                                        wait=max(1, int(min_gap_s / hop_s)),
                                        pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.15)
    return frames * hop / sr_a


def segment_notes(track: PitchTrack, onsets: Optional[np.ndarray] = None,
                  min_note_s: float = 0.05, split_cents: float = 70.0,
                  merge_gap_s: float = 0.02, quantize: bool = False) -> List[Note]:
    """Convierte la trayectoria de f0 en una lista de notas."""
    midi = track.midi
    voiced = track.voiced
    n = len(voiced)
    hop = track.hop_s
    onset_frames = set()
    if onsets is not None:
        for t in np.asarray(onsets):
            fi = int(round(t / hop))
            # ajustar al primer frame sonoro cercano
            for d in (0, 1, 2, -1, 3, -2, 4):
                j = fi + d
                if 0 <= j < n and voiced[j]:
                    onset_frames.add(j)
                    break

    segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        start = i
        ref = [midi[i]]
        j = i + 1
        while j < n and voiced[j]:
            if j in onset_frames and j - start >= 2:
                break
            med = float(np.median(ref[-4:]))
            if abs(midi[j] - med) > split_cents / 100.0:
                # ¿cambio sostenido? (dos frames seguidos)
                if j + 1 < n and voiced[j + 1] and abs(midi[j + 1] - med) > split_cents / 100.0:
                    break
            ref.append(midi[j])
            j += 1
        segments.append((start, j))
        i = j

    rms_ref = np.percentile(track.rms_db[voiced], 95) if voiced.any() else 0.0
    notes: List[Note] = []
    for s, e in segments:
        seg_f0 = track.f0[s:e]
        seg_midi = midi[s:e]
        f0_med = float(np.median(seg_f0))
        m = int(round(float(np.median(seg_midi))))
        r = float(np.max(track.rms_db[s:e]))
        vel = int(np.clip(40 + 87 * (r - (rms_ref - 30.0)) / 30.0, 1, 127))
        conf = float(np.mean(track.confidence[s:e]))
        notes.append(Note(start=s * hop, end=e * hop, midi=m,
                          f0=float(librosa.midi_to_hz(m)) if quantize else f0_med,
                          velocity=vel, confidence=conf))

    # fusionar fragmentos muy cortos con el vecino de igual altura
    merged: List[Note] = []
    for nt in notes:
        if merged and nt.midi == merged[-1].midi and nt.start - merged[-1].end <= merge_gap_s \
                and (nt.duration < min_note_s or merged[-1].duration < min_note_s):
            prev = merged[-1]
            prev.end = nt.end
            prev.velocity = max(prev.velocity, nt.velocity)
            continue
        merged.append(nt)
    notes = [nt for nt in merged if nt.duration >= min_note_s]
    return notes


def notes_to_dicts(notes: List[Note]) -> list[dict]:
    return [nt.to_dict() for nt in notes]


def note_activity(notes: List[Note], n_samples: int, sr: int, release_s: float = 0.03) -> np.ndarray:
    """Envolvente 0..1 que vale 1 mientras suena una nota (con caída corta al final)."""
    act = np.zeros(n_samples, dtype=np.float32)
    rel = max(1, int(release_s * sr))
    ramp = np.linspace(1, 0, rel, dtype=np.float32)
    for nt in notes:
        a, b = int(nt.start * sr), min(n_samples, int(nt.end * sr))
        if b > a:
            act[a:b] = 1.0
            tail = act[b:b + rel]
            act[b:b + rel] = np.maximum(tail, ramp[:len(tail)])
    return act
