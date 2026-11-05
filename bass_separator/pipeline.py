"""Orquestación completa: archivo WAV -> contrabajo aislado y realzado."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import __version__
from .analysis import (AudioAnalysis, analyze_audio, parse_speed_correction, recommend_settings)
from .audio_io import (ensure_stereo, is_effectively_mono, load_audio, match_length, resample,
                       save_audio, speed_correct, to_mono)
from .config import Settings
from .dsp import limiter, match_rms, normalize_peak, rms
from .enhance import enhance_bass
from .midi import write_midi
from .notes import Note, detect_onsets, notes_to_dicts, segment_notes
from .pitch import track_pitch
from .restore import pre_restore
from .separation import DEMUCS_SR, get_separator, pick_device
from .synth import DoubleBassSynth

logger = logging.getLogger(__name__)
ProgressCb = Optional[Callable[[str, float], None]]


@dataclass
class Result:
    input_path: str
    output_path: str
    midi_path: Optional[str] = None
    notes_json_path: Optional[str] = None
    enhanced_path: Optional[str] = None      # solo realce, sin capa sintética (modo hybrid)
    synth_path: Optional[str] = None         # solo capa sintética (modo hybrid)
    raw_path: Optional[str] = None
    stem_paths: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    sample_rate: int = 0
    n_notes: int = 0
    voiced_fraction: float = 0.0
    mono_input: bool = True
    timings: Dict[str, float] = field(default_factory=dict)
    restore: Dict[str, object] = field(default_factory=dict)
    settings: Dict[str, object] = field(default_factory=dict)
    analysis: Dict[str, object] = field(default_factory=dict)
    choices: Dict[str, object] = field(default_factory=dict)   # parámetros elegidos automáticamente
    speed_correction_cents: float = 0.0
    passes: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _notify(cb: ProgressCb, msg: str, frac: float) -> None:
    logger.info("[%3d%%] %s", int(frac * 100), msg)
    if cb is not None:
        cb(msg, frac)


def _analyze_line(y: np.ndarray, sr: int, settings: Settings, beats, key_pcs, device: str):
    """Seguimiento de f0 + ataques + notas con los priores activos."""
    priors = settings.musical_priors
    track = track_pitch(y, sr, fmin=settings.fmin, fmax=settings.fmax, method=settings.pitch_method,
                        interval_prior=priors, key_pcs=key_pcs if priors else None,
                        crepe_model=settings.crepe_model if settings.crepe_model != "auto" else "tiny",
                        device=device)
    onsets = detect_onsets(y, sr, beats=beats if priors else None)
    notes = segment_notes(track, onsets, quantize=settings.quantize_pitch, beats=beats if priors else None)
    return track, onsets, notes


def process_array(y: np.ndarray, sr: int, settings: Settings, progress: ProgressCb = None,
                  separator=None, keep_stems: bool = False, analysis: Optional[AudioAnalysis] = None,
                  explicit=()) -> dict:
    """Procesa audio en memoria. Devuelve un dict con las señales y metadatos.

    Con ``settings.auto`` se analiza primero el audio original y se eligen los
    parámetros (perfil, ensamble/TTA, corrección de velocidad, ruido, graves,
    seguidor de altura) salvo los nombrados en ``explicit``.
    """
    timings: Dict[str, float] = {}
    t0 = time.time()
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    mono_in = is_effectively_mono(y)

    # 0. análisis previo y elección de parámetros
    choices: Dict[str, object] = {}
    if analysis is None and (settings.auto or settings.musical_priors
                             or parse_speed_correction(settings.speed_correction) is None):
        _notify(progress, "Análisis previo de la grabación", 0.01)
        analysis = analyze_audio(y, sr)
    if settings.auto and analysis is not None:
        settings, ch = recommend_settings(analysis, settings, explicit=explicit, device=settings.device)
        choices = {k: {"valor": v, "motivo": why} for k, (v, why) in ch.items()}
        logger.info("Parámetros automáticos: %s", ", ".join(f"{k}={v}" for k, (v, _) in ch.items()))
    if settings.pitch_method == "auto":
        settings = replace(settings, pitch_method="salience")
    timings["analysis"] = time.time() - t0
    beats = analysis.beats.copy() if analysis is not None and len(analysis.beats) else None
    key_pcs = analysis.key_pcs if (analysis is not None and analysis.key_confidence >= 0.3) else None
    device = pick_device(settings.device) if settings.backend == "demucs" else "cpu"

    # 1. restauración previa
    _notify(progress, "Restauración previa", 0.04)
    t1 = time.time()
    y_r, rinfo = pre_restore(y, sr, settings)
    timings["restore"] = time.time() - t1

    # 1b. corrección de velocidad (afinación de referencia A440)
    cents = parse_speed_correction(settings.speed_correction)
    if cents is None:
        cents = analysis.tuning_cents if (analysis is not None and analysis.tuning_confidence >= 0.25
                                          and abs(analysis.tuning_cents) >= 8.0) else 0.0
    if abs(cents) > 0:
        _notify(progress, f"Corrección de velocidad ({cents:+.0f} cents)", 0.06)
        y_r = speed_correct(y_r, sr, cents)
        if beats is not None:
            beats = beats * 2.0 ** (cents / 1200.0)
    n_in = y_r.shape[-1]

    # 2. separación
    _notify(progress, f"Separación de fuentes ({settings.backend})", 0.08)
    t1 = time.time()
    y44 = ensure_stereo(resample(y_r, sr, DEMUCS_SR))
    sep = separator if separator is not None else get_separator(settings, keep_all_stems=keep_stems)
    sep_res = sep.separate(y44, DEMUCS_SR)
    bass = sep_res.stems["bass"]
    timings["separate"] = time.time() - t1

    want_mono = settings.stereo == "mono" or (settings.stereo == "auto" and mono_in)
    bass_proc = to_mono(bass) if want_mono else bass
    n44 = bass_proc.shape[-1]
    duration = n44 / DEMUCS_SR

    # 3. análisis de la línea de bajo (pasada 1, sobre el stem)
    _notify(progress, "Seguimiento de la línea de bajo", 0.6)
    t2 = time.time()
    track, onsets, notes = _analyze_line(bass_proc, DEMUCS_SR, settings, beats, key_pcs, device)
    timings["pitch"] = time.time() - t2
    passes: Dict[str, object] = {"pass1_notes": len(notes), "pass1_voiced": track.voiced_fraction()}

    # 4. realce (y doble pasada: volver a seguir la altura sobre la señal realzada,
    #    más limpia y con la fundamental reconstruida, y realzar de nuevo con esa pista)
    einfo = None
    if settings.mode == "raw":
        out = normalize_peak(bass_proc, settings.target_peak_db)
    else:
        _notify(progress, "Realce del contrabajo", 0.72)
        t3 = time.time()
        out, einfo = enhance_bass(bass_proc, DEMUCS_SR, track, notes, onsets, settings)
        timings["enhance"] = time.time() - t3
        if settings.two_pass:
            _notify(progress, "Segunda pasada de seguimiento y realce", 0.8)
            t3b = time.time()
            track2, onsets2, notes2 = _analyze_line(out, DEMUCS_SR, settings, beats, key_pcs, device)
            if notes2:
                track, onsets, notes = track2, onsets2, notes2
                out, einfo = enhance_bass(bass_proc, DEMUCS_SR, track, notes, onsets, settings)
            passes.update(pass2_notes=len(notes2), pass2_voiced=track2.voiced_fraction())
            timings["second_pass"] = time.time() - t3b
    enhanced = out                      # señal "pura" (sin capa sintética)

    # 5. resíntesis
    synth_layer = None
    if settings.mode in ("hybrid", "resynth") and notes:
        _notify(progress, "Resíntesis del contrabajo", 0.88)
        t4 = time.time()
        synth = DoubleBassSynth(DEMUCS_SR, brightness=settings.synth_brightness)
        synth_layer = match_length(synth.render(notes, duration), n44)
        if settings.mode == "resynth":
            out = synth_layer if out.ndim == 1 else np.repeat(synth_layer[None, :], out.shape[0], axis=0)
        else:
            level = rms(out) * float(np.clip(settings.resynth_mix, 0.0, 2.0))
            layer = match_rms(synth_layer, level)
            out = out + (layer if out.ndim == 1 else layer[None, :])
        out = normalize_peak(out, settings.target_peak_db)
        out = limiter(out, DEMUCS_SR, ceiling_db=settings.target_peak_db)
        timings["resynth"] = time.time() - t4

    # 6. frecuencia de salida
    out_sr = int(settings.output_sr or sr)
    n_out = int(round(n_in * out_sr / sr))

    def finish(sig: np.ndarray) -> np.ndarray:
        return match_length(resample(sig, DEMUCS_SR, out_sr), n_out)

    out = finish(out)
    enhanced = finish(enhanced)
    raw = finish(normalize_peak(bass_proc, settings.target_peak_db))
    synth_alone = finish(normalize_peak(synth_layer, settings.target_peak_db)) if synth_layer is not None else None
    timings["total"] = time.time() - t0
    _notify(progress, "Listo", 1.0)
    return dict(output=out, output_sr=out_sr, enhanced=enhanced, raw=raw, synth=synth_alone,
                stems=sep_res.stems, stems_sr=DEMUCS_SR, notes=notes, track=track, onsets=onsets,
                mono_input=mono_in, restore=asdict(rinfo), enhance=einfo,
                timings=timings, models=sep_res.models, analysis=analysis, choices=choices,
                speed_cents=float(cents), passes=passes, settings=settings)


def process_file(input_path: str | Path, output_dir: str | Path, settings: Settings,
                 progress: ProgressCb = None, separator=None, explicit=()) -> Result:
    """Procesa un archivo y escribe los resultados en ``output_dir``."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_name = input_path.stem
    y, sr = load_audio(input_path)
    logger.info("Entrada: %s (%d canales, %d Hz, %.1f s)", input_path.name, y.shape[0], sr, y.shape[-1] / sr)
    res = process_array(y, sr, settings, progress=progress, separator=separator,
                        keep_stems=settings.keep_stems, explicit=explicit)
    settings = res["settings"]          # parámetros efectivos (tras el análisis automático)

    out_path = output_dir / f"{stem_name}_contrabajo.wav"
    save_audio(out_path, res["output"], res["output_sr"], subtype=settings.bit_depth)
    result = Result(input_path=str(input_path), output_path=str(out_path),
                    duration_s=float(y.shape[-1] / sr), sample_rate=res["output_sr"],
                    n_notes=len(res["notes"]), voiced_fraction=res["track"].voiced_fraction(),
                    mono_input=res["mono_input"], timings=res["timings"], restore=res["restore"],
                    settings=settings.to_dict(),
                    analysis=res["analysis"].to_dict() if res["analysis"] is not None else {},
                    choices=res["choices"], speed_correction_cents=res["speed_cents"],
                    passes=res["passes"])
    if settings.write_midi and res["notes"]:
        midi_path = output_dir / f"{stem_name}_contrabajo.mid"
        write_midi(midi_path, res["notes"])
        result.midi_path = str(midi_path)
    if settings.write_notes_json:
        json_path = output_dir / f"{stem_name}_contrabajo_notas.json"
        meta = dict(version=__version__, input=str(input_path), models=res["models"],
                    duration_s=result.duration_s, n_notes=len(res["notes"]),
                    voiced_fraction=result.voiced_fraction, timings=res["timings"],
                    restore=res["restore"], analysis=result.analysis, choices=result.choices,
                    speed_correction_cents=result.speed_correction_cents, passes=result.passes,
                    settings=settings.to_dict(), notes=notes_to_dicts(res["notes"]))
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        result.notes_json_path = str(json_path)
    if settings.write_layers and settings.mode == "hybrid":
        # la señal realzada sola ("casi pura") y la capa sintética sola, para mezclar a gusto
        enh_path = output_dir / f"{stem_name}_contrabajo_realzado.wav"
        save_audio(enh_path, res["enhanced"], res["output_sr"], subtype=settings.bit_depth)
        result.enhanced_path = str(enh_path)
        if res["synth"] is not None:
            syn_path = output_dir / f"{stem_name}_contrabajo_sintetico.wav"
            save_audio(syn_path, res["synth"], res["output_sr"], subtype=settings.bit_depth)
            result.synth_path = str(syn_path)
    if settings.keep_raw:
        raw_path = output_dir / f"{stem_name}_contrabajo_sin_realce.wav"
        save_audio(raw_path, res["raw"], res["output_sr"], subtype=settings.bit_depth)
        result.raw_path = str(raw_path)
    if settings.keep_stems:
        for name, stem in res["stems"].items():
            if name == "bass":
                continue
            p = output_dir / f"{stem_name}_{name}.wav"
            save_audio(p, resample(stem, res["stems_sr"], res["output_sr"]), res["output_sr"],
                       subtype=settings.bit_depth)
            result.stem_paths[name] = str(p)
    logger.info("Salida: %s (%d notas detectadas)", out_path, result.n_notes)
    return result
