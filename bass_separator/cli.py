"""Interfaz de línea de comandos (``bass-separator``)."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import PRESETS, Settings, apply_preset

SUBCOMMANDS = ("separate", "download-models", "analyze", "demo", "ui")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true", help="mostrar más información")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bass-separator",
        description="Aísla y realza la pista de contrabajo de una grabación de jazz (WAV).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("separate", help="separar y realzar el contrabajo de uno o varios WAV")
    sp.add_argument("inputs", nargs="+", help="archivo(s) WAV de entrada")
    sp.add_argument("-o", "--output", default="salida", help="carpeta de salida (por defecto: salida/)")
    sp.add_argument("--profile", choices=sorted(PRESETS), default="vintage",
                    help="perfil: vintage (78 rpm, big bands antiguas) o modern")
    sp.add_argument("--mode", choices=["raw", "enhance", "hybrid", "resynth"], default="hybrid",
                    help="raw: stem sin tocar; enhance: realce DSP; hybrid: realce + capa resintetizada; "
                         "resynth: solo contrabajo resintetizado")
    sp.add_argument("--backend", choices=["demucs", "dsp"], default="demucs",
                    help="demucs (red neuronal, recomendado) o dsp (sin red, calidad baja)")
    sp.add_argument("--model", default="htdemucs_ft",
                    help="modelo(s) Demucs separados por coma: htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra")
    sp.add_argument("--model-repo", default=None, help="carpeta local con pesos Demucs (uso sin conexión)")
    sp.add_argument("--device", default="auto", help="auto, cpu, cuda o mps")
    sp.add_argument("--shifts", type=int, default=1, help="aumentos en tiempo de prueba (más lento, algo mejor)")
    sp.add_argument("--full-bag", action="store_true",
                    help="usar los 4 sub-modelos de htdemucs_ft aunque solo se necesite el bajo")
    sp.add_argument("--pitch-method", choices=["salience", "pyin"], default="salience")
    sp.add_argument("--fmin", type=float, default=38.0, help="f0 mínima (Hz)")
    sp.add_argument("--fmax", type=float, default=330.0, help="f0 máxima (Hz)")
    sp.add_argument("--resynth-mix", type=float, default=None, help="nivel de la capa resintetizada (0..1)")
    sp.add_argument("--sub-gain", type=float, default=None, help="reconstrucción de la fundamental (0..2)")
    sp.add_argument("--denoise", type=float, default=None, help="reducción de ruido del stem (0..1)")
    sp.add_argument("--harmonic-strength", type=float, default=None, help="fuerza de la máscara armónica (0..1)")
    sp.add_argument("--bleed-reduction", type=float, default=None, help="reducción de colado de batería (0..1)")
    sp.add_argument("--compress", type=float, default=None, help="cantidad de compresión (0..1)")
    sp.add_argument("--body-db", type=float, default=None, help="realce de cuerpo ~90 Hz (dB)")
    sp.add_argument("--presence-db", type=float, default=None, help="realce de presencia ~800 Hz (dB)")
    sp.add_argument("--lowpass", type=float, default=None, help="pasa-bajos final (Hz)")
    sp.add_argument("--no-declick", action="store_true", help="no eliminar clics/crepitar")
    sp.add_argument("--hum-notch", action="store_true", help="eliminar zumbido de red (50/60 Hz) si se detecta")
    sp.add_argument("--quantize-pitch", action="store_true", help="afinar las notas resintetizadas a semitonos")
    sp.add_argument("--stereo", choices=["auto", "mono", "stereo"], default="auto")
    sp.add_argument("--sr", type=int, default=None, help="frecuencia de muestreo de salida (por defecto la de entrada)")
    sp.add_argument("--bit-depth", choices=["PCM_16", "PCM_24", "FLOAT"], default="PCM_24")
    sp.add_argument("--no-midi", action="store_true", help="no escribir el archivo MIDI")
    sp.add_argument("--no-json", action="store_true", help="no escribir el JSON de notas")
    sp.add_argument("--no-layers", action="store_true",
                    help="en modo hybrid, no guardar por separado el realce (sin capa sintética) y la capa sintética")
    sp.add_argument("--keep-raw", action="store_true", help="guardar también el stem de bajo sin realce")
    sp.add_argument("--keep-stems", action="store_true", help="guardar también batería, voces y otros")
    _add_common(sp)

    dp = sub.add_parser("download-models", help="descargar y verificar los pesos oficiales de Demucs")
    dp.add_argument("--model", default="htdemucs_ft", help="modelo(s) separados por coma")
    dp.add_argument("--bass-only", action="store_true", help="solo el sub-modelo de bajo de htdemucs_ft")
    _add_common(dp)

    ap = sub.add_parser("analyze", help="informe rápido de un WAV (mono, ancho de banda, ruido, zumbido)")
    ap.add_argument("input")
    _add_common(ap)

    mp = sub.add_parser("demo", help="generar una canción sintética de prueba (big band degradada)")
    mp.add_argument("-o", "--output", default="demo_bigband.wav")
    mp.add_argument("--duration", type=float, default=20.0)
    mp.add_argument("--seed", type=int, default=0)
    mp.add_argument("--clean", action="store_true", help="sin degradación 78 rpm")
    _add_common(mp)

    up = sub.add_parser("ui", help="interfaz web local (requiere: pip install bass-separator[ui])")
    up.add_argument("--port", type=int, default=7860)
    up.add_argument("--share", action="store_true")
    _add_common(up)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    s = apply_preset(Settings(), args.profile)
    overrides = dict(
        mode=args.mode, backend=args.backend, models=tuple(m.strip() for m in args.model.split(",") if m.strip()),
        device=args.device, shifts=args.shifts, bass_only_models=not args.full_bag,
        pitch_method=args.pitch_method, fmin=args.fmin, fmax=args.fmax,
        quantize_pitch=args.quantize_pitch, stereo=args.stereo, output_sr=args.sr,
        bit_depth=args.bit_depth, write_midi=not args.no_midi, write_notes_json=not args.no_json,
        write_layers=not args.no_layers,
        keep_raw=args.keep_raw, keep_stems=args.keep_stems, hum_notch=args.hum_notch,
    )
    if args.no_declick:
        overrides["declick"] = False
    for attr, val in (("resynth_mix", args.resynth_mix), ("sub_gain", args.sub_gain),
                      ("denoise", args.denoise), ("harmonic_strength", args.harmonic_strength),
                      ("bleed_reduction", args.bleed_reduction), ("compress", args.compress),
                      ("body_db", args.body_db), ("presence_db", args.presence_db),
                      ("lowpass_hz", args.lowpass)):
        if val is not None:
            overrides[attr] = val
    return replace(s, **overrides)


def cmd_separate(args: argparse.Namespace) -> int:
    from .pipeline import process_file
    settings = settings_from_args(args)
    rc = 0
    for inp in args.inputs:
        try:
            res = process_file(inp, args.output, settings)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error("Error procesando %s: %s", inp, exc, exc_info=args.verbose)
            rc = 1
            continue
        print(f"✔ {inp}\n   contrabajo: {res.output_path}")
        if res.enhanced_path:
            print(f"   solo realce (sin capa sintética): {res.enhanced_path}")
        if res.synth_path:
            print(f"   solo capa sintética:              {res.synth_path}")
        if res.midi_path:
            print(f"   MIDI:       {res.midi_path}")
        if res.notes_json_path:
            print(f"   notas:      {res.notes_json_path}")
        print(f"   {res.n_notes} notas, {res.voiced_fraction * 100:.0f} % del tiempo con bajo, "
              f"{res.timings.get('total', 0):.1f} s")
    return rc


def cmd_download(args: argparse.Namespace) -> int:
    from .separation import download_models
    names = [m.strip() for m in args.model.split(",") if m.strip()]
    done = download_models(names, bass_only=args.bass_only)
    print("Modelos verificados y disponibles:", ", ".join(done))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    import numpy as np
    from .audio_io import is_effectively_mono, load_audio, to_mono
    from .dsp import amp2db, rms
    from .restore import detect_hum
    from scipy import signal
    y, sr = load_audio(args.input)
    mono = to_mono(y)
    f, p = signal.welch(mono, fs=sr, nperseg=1 << 14)
    pdb = 10 * np.log10(p + 1e-20)
    peak = pdb.max()
    above = f[pdb > peak - 40]
    print(f"Archivo:        {args.input}")
    print(f"Canales/SR:     {y.shape[0]} / {sr} Hz, duración {y.shape[-1] / sr:.1f} s")
    print(f"Mono efectivo:  {'sí' if is_effectively_mono(y) else 'no'}")
    print(f"Nivel RMS:      {amp2db(rms(mono)):.1f} dBFS, pico {amp2db(np.abs(mono).max()):.1f} dBFS")
    if len(above):
        print(f"Ancho de banda: ~{above.min():.0f} – {above.max():.0f} Hz (a -40 dB del máximo)")
    hum = detect_hum(y, sr)
    print(f"Zumbido de red: {'%.0f Hz' % hum if hum else 'no detectado'}")
    low = pdb[(f >= 40) & (f < 100)].mean()
    mid = pdb[(f >= 100) & (f < 300)].mean()
    print(f"Energía 40-100 Hz relativa a 100-300 Hz: {low - mid:+.1f} dB "
          f"({'fundamentales del bajo débiles' if low - mid < -12 else 'graves presentes'})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .audio_io import save_audio
    from .synthetic import make_song
    song = make_song(duration=args.duration, seed=args.seed, degrade=not args.clean)
    out = Path(args.output)
    save_audio(out, song.mix, song.sr)
    ref = out.with_name(out.stem + "_contrabajo_referencia.wav")
    save_audio(ref, song.bass, song.sr)
    print(f"Demo escrita en {out} (referencia limpia del contrabajo: {ref})")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch
    launch(port=args.port, share=args.share)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # permitir `bass-separator cancion.wav` como atajo de `separate`
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["separate"] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return {"separate": cmd_separate, "download-models": cmd_download, "analyze": cmd_analyze,
            "demo": cmd_demo, "ui": cmd_ui}[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
