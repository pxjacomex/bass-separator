"""Backends de separación de fuentes.

* :class:`DemucsSeparator`: Hybrid Transformer Demucs (Meta AI Research, MIT).
  Los pesos se descargan desde el servidor oficial de Meta
  (``https://dl.fbaipublicfiles.com/demucs/``) y ``torch.hub`` verifica su
  SHA-256 contra el hash incluido en el nombre del archivo (``check_hash=True``),
  por lo que un archivo corrupto o manipulado se rechaza.
* :class:`DSPSeparator`: alternativa sin red neuronal (HPSS + banda del
  contrabajo), útil sin conexión o para comparar. Su calidad es muy inferior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Settings
from .dsp import highpass, lowpass, spectral_gate, stft, istft

logger = logging.getLogger(__name__)

DEMUCS_SR = 44100
# Sub-modelo de htdemucs_ft entrenado específicamente para el bajo (ver
# demucs/remote/htdemucs_ft.yaml: pesos diagonales -> modelo 2 = "bass").
HTDEMUCS_FT_BASS_SIG = "d12395a8"
KNOWN_MODELS = {
    "htdemucs_ft": "Hybrid Transformer Demucs afinado (4 sub-modelos, mejor calidad)",
    "htdemucs": "Hybrid Transformer Demucs (1 modelo, más rápido)",
    "htdemucs_6s": "Variante de 6 fuentes (añade piano y guitarra)",
    "mdx_extra": "Hybrid Demucs entrenado con datos extra (ensamble de 4)",
    "mdx_extra_q": "mdx_extra cuantizado (más ligero)",
}


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SeparationResult:
    stems: Dict[str, np.ndarray]     # cada stem (2, N) a 44.1 kHz
    sr: int
    models: List[str]


class DemucsSeparator:
    """Envuelve ``demucs.apply.apply_model`` con ensamblado de varios modelos."""

    def __init__(self, models: Sequence[str] = ("htdemucs_ft",), device: str = "auto",
                 shifts: int = 1, overlap: float = 0.25, segment: Optional[float] = None,
                 bass_only: bool = True, model_repo: Optional[str | Path] = None,
                 progress: bool = True):
        self.model_names = list(models)
        self.device_name = device
        self.shifts = int(shifts)
        self.overlap = float(overlap)
        self.segment = segment
        self.bass_only = bass_only
        self.model_repo = Path(model_repo) if model_repo else None
        self.progress = progress
        self._loaded = None

    # -- carga -----------------------------------------------------------------
    def resolve_name(self, name: str) -> str:
        """Nombre efectivo del modelo (solo el sub-modelo de bajo si procede)."""
        if name == "htdemucs_ft" and self.bass_only:
            return HTDEMUCS_FT_BASS_SIG
        return name

    def load(self):
        if self._loaded is not None:
            return self._loaded
        from demucs.pretrained import get_model
        loaded = []
        for name in self.model_names:
            eff = self.resolve_name(name)
            logger.info("Cargando modelo Demucs %s%s", name, f" (sub-modelo {eff})" if eff != name else "")
            model = get_model(eff, repo=self.model_repo)
            model.eval()
            loaded.append((name, model))
        self._loaded = loaded
        return loaded

    # -- separación ---------------------------------------------------------------
    def separate(self, audio: np.ndarray, sr: int = DEMUCS_SR) -> SeparationResult:
        """``audio``: (2, N) float32 a 44.1 kHz. Devuelve los stems promediados."""
        import torch
        from demucs.apply import apply_model

        if sr != DEMUCS_SR:
            raise ValueError("DemucsSeparator espera audio a 44100 Hz")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError("DemucsSeparator espera audio estéreo (2, N)")
        device = pick_device(self.device_name)
        models = self.load()
        wav = torch.from_numpy(audio)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std

        acc: Dict[str, List[np.ndarray]] = {}
        for name, model in models:
            kwargs = dict(shifts=self.shifts, split=True, overlap=self.overlap,
                          progress=self.progress, device=device)
            if self.segment is not None:
                kwargs["segment"] = float(self.segment)
            with torch.no_grad():
                out = apply_model(model, wav[None], **kwargs)[0]
            out = out * std + mean
            for i, src in enumerate(model.sources):
                acc.setdefault(src, []).append(out[i].cpu().numpy().astype(np.float32))
        stems = {k: np.mean(np.stack(v), axis=0) for k, v in acc.items()}
        if "bass" not in stems:
            raise RuntimeError("El modelo no produce la fuente 'bass'")
        return SeparationResult(stems=stems, sr=DEMUCS_SR, models=self.model_names)


class DSPSeparator:
    """Separación aproximada sin red neuronal (solo para pruebas o sin conexión)."""

    def __init__(self, lowpass_hz: float = 2500.0, highpass_hz: float = 30.0, denoise: float = 0.3):
        self.lowpass_hz = lowpass_hz
        self.highpass_hz = highpass_hz
        self.denoise = denoise

    def separate(self, audio: np.ndarray, sr: int = DEMUCS_SR) -> SeparationResult:
        import librosa
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        out = np.empty_like(audio)
        for c in range(audio.shape[0]):
            y = audio[c]
            S = stft(y, 2048, 512)
            H, _ = librosa.decompose.hpss(S, kernel_size=(17, 17), margin=(1.0, 1.0))
            h = istft(H, 512, len(y))
            h = highpass(h, sr, self.highpass_hz, order=2)
            h = lowpass(h, sr, self.lowpass_hz, order=2)
            out[c] = spectral_gate(h, sr, self.denoise) if self.denoise > 0 else h
        if out.shape[0] == 1:
            out = np.repeat(out, 2, axis=0)
        return SeparationResult(stems={"bass": out}, sr=sr, models=["dsp"])


def get_separator(settings: Settings, keep_all_stems: bool = False, model_repo=None, progress: bool = True):
    if settings.backend == "dsp":
        return DSPSeparator()
    if settings.backend == "demucs":
        return DemucsSeparator(models=settings.models, device=settings.device, shifts=settings.shifts,
                               overlap=settings.overlap, segment=settings.segment,
                               bass_only=settings.bass_only_models and not keep_all_stems,
                               model_repo=model_repo, progress=progress)
    raise ValueError(f"Backend desconocido: {settings.backend!r}")


def download_models(names: Sequence[str] = ("htdemucs_ft",), bass_only: bool = False) -> List[str]:
    """Descarga (y verifica) los pesos oficiales de Demucs en la caché de torch.hub."""
    import torch
    from demucs.pretrained import get_model
    done = []
    for name in names:
        eff = HTDEMUCS_FT_BASS_SIG if (name == "htdemucs_ft" and bass_only) else name
        get_model(eff)
        done.append(eff)
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    logger.info("Pesos disponibles en %s", cache)
    return done
