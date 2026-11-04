import numpy as np
import pytest

torch = pytest.importorskip("torch")
demucs = pytest.importorskip("demucs")

from bass_separator import config  # noqa: E402
from bass_separator.separation import DemucsSeparator, get_separator, HTDEMUCS_FT_BASS_SIG  # noqa: E402


def test_demucs_plumbing_with_untrained_model():
    """Usa el modelo de prueba de demucs (sin pesos, sin red) para validar el flujo."""
    sep = DemucsSeparator(models=["demucs_unittest"], device="cpu", shifts=0, progress=False)
    audio = np.random.default_rng(0).normal(size=(2, 44100 * 3)).astype(np.float32) * 0.1
    res = sep.separate(audio, 44100)
    assert "bass" in res.stems
    assert res.stems["bass"].shape == audio.shape
    assert np.isfinite(res.stems["bass"]).all()


def test_bass_only_resolution():
    sep = DemucsSeparator(models=["htdemucs_ft"], bass_only=True)
    assert sep.resolve_name("htdemucs_ft") == HTDEMUCS_FT_BASS_SIG
    sep = DemucsSeparator(models=["htdemucs_ft"], bass_only=False)
    assert sep.resolve_name("htdemucs_ft") == "htdemucs_ft"
    s = config.Settings(keep_stems=True)
    assert get_separator(s, keep_all_stems=True).bass_only is False


def test_htdemucs_ft_bag_bass_submodel_is_second():
    """Comprueba contra el yaml de demucs que el 2.º sub-modelo es el de 'bass'."""
    from pathlib import Path
    import yaml
    import demucs
    bag = yaml.safe_load((Path(demucs.__file__).parent / "remote" / "htdemucs_ft.yaml").read_text())
    from demucs.pretrained import SOURCES
    assert bag["models"][SOURCES.index("bass")] == HTDEMUCS_FT_BASS_SIG
    w = np.array(bag["weights"])
    assert np.allclose(w, np.eye(4))
