import json
from pathlib import Path

import numpy as np
import pytest

from bass_separator import config, restore
from bass_separator.audio_io import load_audio, save_audio
from bass_separator.pipeline import process_file


def test_declick_removes_impulses(song):
    y = song.bass.copy()
    rng = np.random.default_rng(0)
    pos = rng.integers(1000, len(y) - 1000, 40)
    y[pos] += 0.8
    out, n = restore.declick(y, song.sr)
    assert n >= 30
    assert np.abs(out[pos] - song.bass[pos]).max() < 0.2


def test_hum_detection():
    sr = 22050
    t = np.arange(sr * 25) / sr
    rng = np.random.default_rng(1)
    clean = 0.05 * rng.normal(size=len(t)).astype(np.float32)
    assert restore.detect_hum(clean, sr) == 0.0
    hum = clean + 0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    assert restore.detect_hum(hum, sr) == 60.0


@pytest.mark.parametrize("mode", ["hybrid", "enhance", "resynth", "raw"])
def test_pipeline_dsp_backend(tmp_path: Path, song, mode):
    inp = tmp_path / "mezcla.wav"
    save_audio(inp, song.mix, song.sr)
    s = config.Settings(backend="dsp", mode=mode, keep_raw=True)
    res = process_file(inp, tmp_path / "out", s)
    out, sr = load_audio(res.output_path)
    assert sr == song.sr
    assert out.shape[-1] == len(song.mix)
    assert np.isfinite(out).all()
    assert res.n_notes > 10
    assert Path(res.raw_path).exists()
    if mode == "hybrid":
        # realce solo (sin capa sintética) y capa sintética sola, misma longitud que la mezcla
        enh, _ = load_audio(res.enhanced_path)
        syn, _ = load_audio(res.synth_path)
        assert enh.shape[-1] == out.shape[-1] == syn.shape[-1]
        assert not np.allclose(enh, out)
        assert np.abs(enh).max() <= 10 ** (-1 / 20) + 1e-3
    else:
        assert res.enhanced_path is None and res.synth_path is None
    meta = json.loads(Path(res.notes_json_path).read_text(encoding="utf-8"))
    assert meta["n_notes"] == res.n_notes
    assert Path(res.midi_path).exists()


def test_pipeline_stereo_input_and_output_sr(tmp_path: Path, song):
    inp = tmp_path / "estereo.wav"
    st = np.stack([song.mix, np.roll(song.mix, 5)])
    save_audio(inp, st, song.sr)
    s = config.Settings(backend="dsp", mode="hybrid", output_sr=22050, stereo="auto")
    res = process_file(inp, tmp_path / "out", s)
    out, sr = load_audio(res.output_path)
    assert sr == 22050
    assert out.shape[0] == 2
    enh, sr2 = load_audio(res.enhanced_path)
    assert sr2 == 22050 and enh.shape == out.shape
