"""Interfaz web local con Gradio (dependencia opcional: ``pip install bass-separator[ui]``)."""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from .config import PRESETS, Settings, apply_preset


def _process(file_path, auto, quality, profile, mode, backend, model, resynth_mix, sub_gain, denoise,
             harmonic_strength, shifts, progress=None):
    from .pipeline import process_file
    if not file_path:
        raise ValueError("Sube un archivo WAV")
    if auto:
        settings = replace(Settings(), auto=True, quality=quality, mode=mode, backend=backend,
                           resynth_mix=float(resynth_mix), keep_raw=True)
        explicit = {"resynth_mix"}
    else:
        settings = apply_preset(Settings(), profile)
        settings = replace(settings, auto=False, mode=mode, backend=backend, models=(model,),
                           shifts=int(shifts), resynth_mix=float(resynth_mix), sub_gain=float(sub_gain),
                           denoise=float(denoise), harmonic_strength=float(harmonic_strength),
                           keep_raw=True)
        explicit = set()
    out_dir = Path(tempfile.mkdtemp(prefix="bass_separator_"))

    def cb(msg, frac):
        if progress is not None:
            progress(frac, desc=msg)

    res = process_file(file_path, out_dir, settings, progress=cb, explicit=explicit)
    summary = (f"{res.n_notes} notas detectadas · {res.voiced_fraction * 100:.0f} % del tiempo con bajo · "
               f"{res.timings.get('total', 0):.1f} s de proceso")
    if res.choices:
        summary += "\nParámetros automáticos: " + ", ".join(f"{k}={v['valor']}" for k, v in res.choices.items())
    if res.speed_correction_cents:
        summary += f"\nCorrección de velocidad: {res.speed_correction_cents:+.1f} cents"
    return (res.output_path, res.enhanced_path, res.synth_path, res.raw_path, res.midi_path,
            res.notes_json_path, summary)


def build_app():
    import gradio as gr

    with gr.Blocks(title="bass-separator") as app:
        gr.Markdown("# bass-separator\nAísla y realza el contrabajo de una grabación de jazz.")
        with gr.Row():
            with gr.Column():
                inp = gr.Audio(label="Archivo WAV", type="filepath")
                auto = gr.Checkbox(value=True, label="Automático: analizar la grabación y elegir parámetros")
                quality = gr.Dropdown(["fast", "balanced", "best"], value="balanced", label="Calidad (ensamble y TTA)")
                profile = gr.Dropdown(sorted(PRESETS), value="vintage", label="Perfil (modo manual)")
                mode = gr.Radio(["hybrid", "enhance", "resynth", "raw"], value="hybrid", label="Modo")
                backend = gr.Radio(["demucs", "dsp"], value="demucs", label="Separador")
                model = gr.Dropdown(["htdemucs_ft", "htdemucs", "htdemucs_6s", "mdx_extra"],
                                    value="htdemucs_ft", label="Modelo Demucs")
                shifts = gr.Slider(1, 5, value=1, step=1, label="Shifts (calidad/tiempo)")
                resynth_mix = gr.Slider(0, 1, value=0.35, step=0.05, label="Capa resintetizada")
                sub_gain = gr.Slider(0, 2, value=1.0, step=0.1, label="Reconstrucción de graves")
                denoise = gr.Slider(0, 1, value=0.7, step=0.05, label="Reducción de ruido")
                harmonic_strength = gr.Slider(0, 1, value=0.8, step=0.05, label="Máscara armónica")
                btn = gr.Button("Procesar", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Contrabajo (mezcla final)", type="filepath")
                enh = gr.Audio(label="Solo realce, sin capa sintética (modo hybrid)", type="filepath")
                syn = gr.Audio(label="Solo capa sintética (modo hybrid)", type="filepath")
                raw = gr.Audio(label="Stem sin realce", type="filepath")
                midi = gr.File(label="MIDI de la línea de bajo")
                notes = gr.File(label="Notas (JSON)")
                summary = gr.Textbox(label="Resumen")
        btn.click(lambda *a, progress=gr.Progress(): _process(*a, progress=progress),
                  inputs=[inp, auto, quality, profile, mode, backend, model, resynth_mix, sub_gain, denoise,
                          harmonic_strength, shifts],
                  outputs=[out, enh, syn, raw, midi, notes, summary])
    return app


def launch(port: int = 7860, share: bool = False) -> None:
    try:
        import gradio  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Gradio no está instalado. Ejecuta: pip install 'bass-separator[ui]'") from exc
    build_app().launch(server_port=port, share=share)
