import numpy as np

from bass_separator import config, enhance, notes as nm, pitch, synth
from bass_separator.notes import Note
from tests.conftest import si_sdr


def test_enhance_improves_sdr_on_hard_stem(song, hard_stem):
    ref = song.bass[:len(hard_stem)]
    tr = pitch.track_pitch(hard_stem, song.sr)
    ons = nm.detect_onsets(hard_stem, song.sr)
    nts = nm.segment_notes(tr, ons)
    out, info = enhance.enhance_bass(hard_stem, song.sr, tr, nts, ons, config.Settings())
    assert out.shape == hard_stem.shape
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-3
    assert si_sdr(ref, out) > si_sdr(ref, hard_stem) + 1.0
    assert info.voiced_fraction > 0.7


def test_enhance_stereo(song, hard_stem):
    st = np.stack([hard_stem, hard_stem * 0.9])
    tr = pitch.track_pitch(st, song.sr)
    ons = nm.detect_onsets(st, song.sr)
    nts = nm.segment_notes(tr, ons)
    out, _ = enhance.enhance_bass(st, song.sr, tr, nts, ons, config.Settings())
    assert out.shape == st.shape


def test_synth_pitch_and_level():
    sr = 44100
    s = synth.DoubleBassSynth(sr)
    nts = [Note(start=0.1 + i * 0.5, end=0.55 + i * 0.5, midi=m, f0=440 * 2 ** ((m - 69) / 12),
                velocity=100, confidence=1.0) for i, m in enumerate([28, 33, 38, 43])]
    y = s.render(nts, 2.5)
    assert y.shape[0] == int(np.ceil(2.5 * sr)) + 1
    assert 0.5 < np.abs(y).max() <= 1.0
    tr = pitch.track_pitch(y, sr)
    for n in nts:
        i0, i1 = int((n.start + 0.05) / tr.hop_s), int((n.end - 0.05) / tr.hop_s)
        f = tr.f0[i0:i1]
        f = f[np.isfinite(f)]
        assert len(f) > 0
        assert abs(1200 * np.log2(np.median(f) / n.f0)) < 30
