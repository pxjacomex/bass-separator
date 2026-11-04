import numpy as np
import pytest

from bass_separator import pitch, synthetic
from tests.conftest import pitch_accuracy


def test_salience_clean(song):
    tr = pitch.track_pitch(song.bass, song.sr, method="salience")
    assert pitch_accuracy(tr, song.notes) > 0.95
    assert 0.8 < tr.voiced_fraction() <= 1.0


def test_salience_bandlimited_missing_fundamental(song):
    deg = synthetic.degrade_78rpm(song.bass, song.sr, noise_db=-45, lo=110)
    tr = pitch.track_pitch(deg, song.sr, method="salience")
    assert pitch_accuracy(tr, song.notes) > 0.9


def test_salience_with_bleed(hard_stem, song):
    # colado de banda al 35 % y ruido a -28 dB: caso duro, por encima de pyin
    tr = pitch.track_pitch(hard_stem, song.sr, method="salience")
    assert pitch_accuracy(tr, song.notes) > 0.75
    assert pitch_accuracy(tr, song.notes) > pitch_accuracy(
        pitch.track_pitch(hard_stem, song.sr, method="pyin"), song.notes)


def test_pyin_clean(song):
    tr = pitch.track_pitch(song.bass, song.sr, method="pyin")
    assert pitch_accuracy(tr, song.notes) > 0.85


def test_silence_is_unvoiced():
    y = np.zeros(44100 * 2, dtype=np.float32)
    tr = pitch.track_pitch(y, 44100)
    assert tr.voiced_fraction() == 0.0


def test_f0_at_interpolates(song):
    tr = pitch.track_pitch(song.bass, song.sr)
    t = np.array([0.2, 0.5, 1.0])
    f = tr.f0_at(t)
    assert f.shape == (3,)
    assert np.isfinite(f).all()


def test_unknown_method():
    with pytest.raises(ValueError):
        pitch.track_pitch(np.zeros(1000, dtype=np.float32), 44100, method="nope")
