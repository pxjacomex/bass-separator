import numpy as np
import pytest

from bass_separator import synthetic
from bass_separator.dsp import normalize_peak


@pytest.fixture(scope="session")
def song():
    return synthetic.make_song(duration=12.0, seed=0)


@pytest.fixture(scope="session")
def hard_stem(song):
    """Stem de bajo con colado de banda y ruido (simula la salida de Demucs en 78 rpm)."""
    return synthetic.degrade_78rpm(normalize_peak(song.bass, -3) + 0.35 * normalize_peak(song.band, -3),
                                   song.sr, noise_db=-28)


def si_sdr(ref, est):
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s = a * ref
    e = est - s
    return 10 * np.log10(np.sum(s ** 2) / (np.sum(e ** 2) + 1e-12))


def pitch_accuracy(track, ref_notes, tol_cents=50.0):
    ok = tot = 0
    for nt in ref_notes:
        i0 = int((nt.start + 0.03) / track.hop_s)
        i1 = int((nt.end - 0.02) / track.hop_s)
        for i in range(i0, min(i1, len(track.f0))):
            tot += 1
            if track.voiced[i] and abs(1200 * np.log2(track.f0[i] / nt.f0)) < tol_cents:
                ok += 1
    return ok / max(1, tot)
