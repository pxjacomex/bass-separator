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
    sp.add_argument("--profile", choices=sorted(PRESETS), default=None,
                    help="perfil: vintage (78 rpm, big bands antiguas) o modern (por defecto se elige "
                         "con el análisis previo)")
    sp.add_argument("--no-auto", action="store_true",
                    help="no elegir parámetros automáticamente a partir del análisis previo")
    sp.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced",
                    help="estrategia de separación: fast (1 modelo, 1 shift), balanced (según dispositivo), "
                         "best (ensamble htdemucs_ft+mdx_extra, 3 shifts)")
    sp.add_argument("--speed-correction", default="auto",
                    help="auto | off | cents (p. ej. +30): corregir la velocidad/afinación respecto a A440")
    sp.add_argument("--no-two-pass", action="store_true",
                    help="no repetir el seguimiento de altura sobre la señal realzada")
    sp.add_argument("--no-priors", action="store_true",
                    help="desactivar los priores musicales (intervalos, tonalidad, pulso)")
    sp.add_argument("--mode", choices=["raw", "enhance", "hybrid", "resynth"], default="hybrid",
                    help="raw: stem sin tocar; enhance: realce DSP; hybrid: realce + capa resintetizada; "
                         "resynth: solo contrabajo resintetizado")
    sp.add_argument("--backend", choices=["demucs", "dsp"], default="demucs",
                    help="demucs (red neuronal, recomendado) o dsp (sin red, calidad baja)")
    sp.add_argument("--model", default=None,
                    help="modelo(s) Demucs separados por coma: htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra "
                         "(por defecto según --quality y el dispositivo)")
    sp.add_argument("--model-repo", default=None, help="carpeta local con pesos Demucs (uso sin conexión)")
    sp.add_argument("--device", default="auto", help="auto, cpu, cuda o mps")
    sp.add_argument("--shifts", type=int, default=None,
                    help="aumentos en tiempo de prueba (más lento, algo mejor; por defecto según --quality)")
    sp.add_argument("--full-bag", action="store_true",
                    help="usar los 4 sub-modelos de htdemucs_ft aunque solo se necesite el bajo")
    sp.add_argument("--pitch-method", choices=["auto", "salience", "fusion", "crepe", "pyin"], default="auto",
                    help="seguidor de f0: salience (propio), fusion (salience + CREPE), crepe, pyin")
    sp.add_argument("--crepe-model", choices=["auto", "tiny", "full"], default="auto",
                    help="tamaño del modelo CREPE (full es mucho más lento en CPU)")
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

    ap = sub.add_parser("analyze", help="análisis previo de un WAV y parámetros que se elegirían")
    ap.add_argument("input")
    ap.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced")
    ap.add_argument("--device", default="auto")
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


def settings_from_args(args: argparse.Namespace):
    """Devuelve ``(settings, explicit)``: ``explicit`` son los parámetros fijados por el usuario,
    que el análisis automático no debe tocar."""
    explicit = set()
    s = Settings()
    if args.profile:
        s = apply_preset(s, args.profile)
        explicit.add("profile")
        explicit.update(PRESETS[args.profile].keys())
    overrides = dict(
        mode=args.mode, backend=args.backend, device=args.device, bass_only_models=not args.full_bag,
        auto=not args.no_auto, quality=args.quality, speed_correction=args.speed_correction,
        two_pass=not args.no_two_pass, musical_priors=not args.no_priors,
        pitch_method=args.pitch_method, crepe_model=args.crepe_model, fmin=args.fmin, fmax=args.fmax,
        quantize_pitch=args.quantize_pitch, stereo=args.stereo, output_sr=args.sr,
        bit_depth=args.bit_depth, write_midi=not args.no_midi, write_notes_json=not args.no_json,
        write_layers=not args.no_layers, keep_raw=args.keep_raw, keep_stems=args.keep_stems,
    )
    if args.model:
        overrides["models"] = tuple(m.strip() for m in args.model.split(",") if m.strip())
        explicit.add("models")
    if args.shifts is not None:
        overrides["shifts"] = args.shifts
        explicit.add("shifts")
    if args.hum_notch:
        overrides["hum_notch"] = True
        explicit.add("hum_notch")
    if args.no_declick:
        overrides["declick"] = False
        explicit.add("declick")
    if args.pitch_method != "auto":
        explicit.add("pitch_method")
    if args.crepe_model != "auto":
        explicit.add("crepe_model")
    if str(args.speed_correction).lower() != "auto":
        explicit.add("speed_correction")
    for attr, val in (("resynth_mix", args.resynth_mix), ("sub_gain", args.sub_gain),
                      ("denoise", args.denoise), ("harmonic_strength", args.harmonic_strength),
                      ("bleed_reduction", args.bleed_reduction), ("compress", args.compress),
                      ("body_db", args.body_db), ("presence_db", args.presence_db),
                      ("lowpass_hz", args.lowpass)):
        if val is not None:
            overrides[attr] = val
            explicit.add(attr)
    return replace(s, **overrides), explicit


def _print_auto_report(res) -> None:
    an = res.analysis
    if an:
        print("   Análisis previo: "
              f"{'mono' if an.get('mono') else 'estéreo'}, banda ~{an.get('band_lo_hz', 0):.0f}-{an.get('band_hi_hz', 0):.0f} Hz, "
              f"SNR {an.get('snr_db', 0):.0f} dB, {an.get('crackle_per_s', 0):.1f} clics/s, "
              f"afinación {an.get('tuning_cents', 0):+.0f} cents, tempo ~{an.get('tempo_bpm', 0):.0f} BPM, "
              f"tonalidad {an.get('key', '?')}, vintage {an.get('vintage_score', 0):.2f}")
    if res.choices:
        print("   Parámetros elegidos: " + ", ".join(f"{k}={v['valor']}" for k, v in res.choices.items()))
    if res.speed_correction_cents:
        print(f"   Corrección de velocidad aplicada: {res.speed_correction_cents:+.1f} cents")
    if res.passes.get("pass2_notes") is not None:
        print(f"   Doble pasada: {res.passes['pass1_notes']} notas en la 1.ª, {res.passes['pass2_notes']} en la 2.ª")


def cmd_separate(args: argparse.Namespace) -> int:
    from .pipeline import process_file
    settings, explicit = settings_from_args(args)
    rc = 0
    for inp in args.inputs:
        try:
            res = process_file(inp, args.output, settings, explicit=explicit)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error("Error procesando %s: %s", inp, exc, exc_info=args.verbose)
            rc = 1
            continue
        print(f"✔ {inp}")
        _print_auto_report(res)
        print(f"   contrabajo: {res.output_path}")
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
    from .analysis import analyze_audio, describe_choices, recommend_settings
    from .audio_io import load_audio
    y, sr = load_audio(args.input)
    an = analyze_audio(y, sr)
    print(f"Archivo: {args.input}")
    print(an.report())
    s, choices = recommend_settings(an, Settings(quality=args.quality), device=args.device)
    print(f"\nParámetros recomendados (--quality {args.quality}, dispositivo {args.device}):")
    print(describe_choices(choices))
    print(f"\nComando equivalente:\n  bass-separator separate \"{args.input}\" --quality {args.quality}")
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
