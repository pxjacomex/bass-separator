import numpy as np
import librosa

from bass_separator import dsp


def test_block_stft_matches_librosa():
    sr = 11025
    t = np.arange(sr * 2) / sr
    y = (0.3 * np.sin(2 * np.pi * 110 * t) + 0.01 * np.random.default_rng(0).normal(size=len(t))).astype(np.float32)
    a = dsp.block_stft_mag(y, 4096, 110, 2048, 500, block_frames=64)
    b = np.abs(librosa.stft(y, n_fft=4096, hop_length=110, win_length=2048))[:500]
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-3


def test_limiter_respects_ceiling():
    sr = 44100
    y = np.random.default_rng(1).normal(size=sr).astype(np.float32) * 3
    out = dsp.limiter(y, sr, ceiling_db=-1.0)
    assert np.abs(out).max() <= dsp.db2amp(-1.0) + 1e-4


def test_filters_preserve_shape_and_level():
    sr = 44100
    y = np.random.default_rng(2).normal(size=(2, sr)).astype(np.float32)
    for fn in (lambda x: dsp.highpass(x, sr, 30), lambda x: dsp.lowpass(x, sr, 5000),
               lambda x: dsp.peaking_eq(x, sr, 800, 2.0), lambda x: dsp.low_shelf(x, sr, 90, 2.0)):
        out = fn(y)
        assert out.shape == y.shape
        assert np.isfinite(out).all()
        assert 0.3 < dsp.rms(out) / dsp.rms(y) < 1.5


def test_spectral_gate_removes_noise_keeps_tone():
    sr = 22050
    t = np.arange(sr * 4) / sr
    rng = np.random.default_rng(3)
    tone = 0.3 * np.sin(2 * np.pi * 98 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)  # nota intermitente
    noise = 0.01 * rng.normal(size=len(t))
    y = (tone + noise).astype(np.float32)
    out = dsp.spectral_gate(y, sr, 0.8)
    silent = np.sin(2 * np.pi * 0.5 * t) <= 0
    assert dsp.rms(out[silent]) < 0.5 * dsp.rms(y[silent])          # ruido reducido
    assert dsp.rms(out[~silent]) > 0.8 * dsp.rms(y[~silent])        # tono conservado
