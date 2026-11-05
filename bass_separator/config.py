"""Parámetros de procesamiento y perfiles predefinidos."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional, Tuple


@dataclass
class Settings:
    """Todos los parámetros ajustables del pipeline.

    Los valores por defecto corresponden al perfil ``vintage`` (grabaciones
    antiguas de big band). Use :func:`apply_preset` para cambiar de perfil.
    """

    # --- Separación de fuentes ------------------------------------------------
    backend: str = "demucs"                    # "demucs" | "dsp"
    models: Tuple[str, ...] = ("htdemucs_ft",)  # modelos Demucs a promediar
    device: str = "auto"                       # "auto" | "cpu" | "cuda" | "mps"
    shifts: int = 1                            # aumentos en tiempo de prueba (TTA)
    overlap: float = 0.25                      # solapamiento entre segmentos
    segment: Optional[float] = None            # segundos por segmento (None = por defecto del modelo)
    bass_only_models: bool = True              # usar solo el sub-modelo de bajo de htdemucs_ft

    # --- Restauración previa (antes de separar) --------------------------------
    profile: str = "vintage"                   # "vintage" | "modern"
    declick: bool = True                       # eliminar clics/crepitar de 78 rpm
    pre_denoise: float = 0.5                   # 0..1 reducción de siseo antes de separar
    hum_notch: bool = False                    # notch 50/60 Hz y armónicos

    # --- Seguimiento de altura --------------------------------------------------
    pitch_method: str = "salience"             # "salience" | "pyin"
    fmin: float = 38.0                         # Hz (E1 = 41.2 Hz)
    fmax: float = 330.0                        # Hz (E4 = 329.6 Hz)

    # --- Realce ---------------------------------------------------------------
    mode: str = "hybrid"                       # "raw" | "enhance" | "hybrid" | "resynth"
    denoise: float = 0.7                       # 0..1 puerta espectral sobre el stem de bajo
    bleed_reduction: float = 0.6               # 0..1 reducción de colado de batería (HPSS)
    harmonic_strength: float = 0.8             # 0..1 cuánto se atenúa lo no armónico
    sub_gain: float = 1.0                      # 0..2 reconstrucción de la fundamental perdida
    highpass_hz: float = 28.0
    lowpass_hz: float = 5000.0
    body_db: float = 1.5                       # realce de cuerpo (~90 Hz)
    presence_db: float = 1.0                   # definición de cuerda (~800 Hz)
    compress: float = 0.5                      # 0..1 cantidad de compresión
    target_peak_db: float = -1.0

    # --- Resíntesis (modelo físico) ---------------------------------------------
    resynth_mix: float = 0.35                  # nivel de la capa resintetizada (relativo)
    quantize_pitch: bool = False               # afinar las notas a semitonos exactos
    synth_brightness: float = 0.5              # 0..1

    # --- Salida ---------------------------------------------------------------
    output_sr: Optional[int] = None            # None = misma frecuencia que la entrada
    stereo: str = "auto"                       # "auto" | "mono" | "stereo"
    write_midi: bool = True
    write_notes_json: bool = True
    write_layers: bool = True                  # en modo hybrid: guardar también realce y capa sintética por separado
    keep_raw: bool = False                     # guardar el stem de bajo sin realce
    keep_stems: bool = False                   # guardar también batería/piano/otros
    bit_depth: str = "PCM_24"                  # subtipo de libsndfile

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # Grabaciones de 78 rpm / transferencias antiguas (Basie, Goodman, Ellington...)
    "vintage": dict(
        profile="vintage", declick=True, pre_denoise=0.5, denoise=0.7,
        bleed_reduction=0.6, harmonic_strength=0.8, sub_gain=1.0,
        lowpass_hz=4500.0, body_db=2.0, presence_db=1.5, compress=0.5,
        resynth_mix=0.35,
    ),
    # Grabaciones modernas de buena calidad
    "modern": dict(
        profile="modern", declick=False, pre_denoise=0.0, denoise=0.3,
        bleed_reduction=0.4, harmonic_strength=0.5, sub_gain=0.4,
        lowpass_hz=9000.0, body_db=0.0, presence_db=0.0, compress=0.3,
        resynth_mix=0.15,
    ),
}


def apply_preset(settings: Settings, name: str) -> Settings:
    """Devuelve una copia de ``settings`` con los valores del perfil ``name``."""
    if name not in PRESETS:
        raise ValueError(f"Perfil desconocido: {name!r}. Opciones: {sorted(PRESETS)}")
    return replace(settings, **PRESETS[name])


__all__ = ["Settings", "PRESETS", "apply_preset"]
