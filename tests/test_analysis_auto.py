"""Análisis previo, corrección de velocidad, priores musicales y doble pasada."""
from pathlib import Path
import json

import numpy as np
import pytest

from bass_separator import analysis, config, enhance, notes as nm, pitch, synthetic
from bass_separator.audio_io import load_audio, resample, save_audio, speed_correct
from bass_separator.dsp import normalize_peak
from bass_separator.pipeline import process_file
from tests.conftest import pitch_accuracy


def _detune(y, sr, cents):
    """Simula una transferencia que suena ``cents`` más aguda (va más rápida)."""
    return resample(y, sr, int(round(sr / 2 ** (cents / 1200))))


@pytest.mark.parametrize("cents", [0.0, 30.0, -25.0])
def test_tuning_estimation(song, cents):
    y = _detune(song.mix, song.sr, cents) if cents else song.mix
    est, conf = analysis.estimate_tuning_cents(y, song.sr)
    assert abs(est - cents) < 4.0
    assert conf > 0.5


def test_speed_correct_restores_pitch(song):
    y = _detune(song.bass, song.sr, 30.0)
    fixed = speed_correct(y, song.sr, 30.0)
    assert abs(len(fixed) - len(song.bass)) < 0.001 * len(song.bass)
    tr = pitch.track_pitch(fixed, song.sr)
    assert pitch_accuracy(tr, song.notes, tol_cents=25.0) > 0.9
    tr_bad = pitch.track_pitch(y, song.sr)
    assert pitch_accuracy(tr_bad, song.notes, tol_cents=25.0) < 0.5


def test_analyze_audio_vintage_vs_clean(song):
    an = analysis.analyze_audio(song.mix, song.sr)
    assert an.vintage_score >= 0.5
    assert an.mono
    assert an.crackle_per_s > 1.0
    assert 70 < an.tempo_bpm < 340            # 160 BPM o su mitad
    assert len(an.beats) > 5
    assert an.key != "?"
    assert "Afinación" in an.report()
    clean = synthetic.make_song(duration=12.0, seed=0, degrade=False)
    an2 = analysis.analyze_audio(clean.mix, clean.sr)
    assert an2.vintage_score < 0.5
    assert an2.snr_db > an.snr_db
    json.dumps(an.to_dict())                  # serializable


def test_recommend_settings_respects_explicit(song):
    an = analysis.analyze_audio(song.mix, song.sr)
    base = config.Settings(sub_gain=1.7, quality="fast")
    s, choices = analysis.recommend_settings(an, base, explicit={"sub_gain"}, device="cpu")
    assert s.sub_gain == 1.7 and "sub_gain" not in choices
    assert s.profile == "vintage" and s.models == ("htdemucs_ft",) and s.shifts == 1
    s_best, _ = analysis.recommend_settings(an, config.Settings(quality="best"), device="cpu")
    assert "mdx_extra" in s_best.models and s_best.shifts == 3
    s_gpu, _ = analysis.recommend_settings(an, config.Settings(), device="cuda")
    assert len(s_gpu.models) == 2 and s_gpu.shifts == 2
    assert s.pitch_method == "salience"
    json.dumps({k: v for k, (v, _) in choices.items()})


def test_speed_correction_recommended_when_detuned(song):
    y = _detune(song.mix, song.sr, 30.0)
    an = analysis.analyze_audio(y, song.sr)
    s, choices = analysis.recommend_settings(an, config.Settings(), device="cpu")
    assert abs(analysis.parse_speed_correction(s.speed_correction) - 30.0) < 4.0
    s2, _ = analysis.recommend_settings(an, config.Settings(speed_correction="off"), device="cpu")
    assert analysis.parse_speed_correction(s2.speed_correction) == 0.0
    assert analysis.parse_speed_correction("+12.5") == 12.5
    assert analysis.parse_speed_correction("auto") is None


def test_musical_priors_do_not_hurt(song, hard_stem):
    key_pcs = analysis.estimate_key(song.mix, song.sr)[2]
    base = pitch_accuracy(pitch.track_pitch(hard_stem, song.sr), song.notes)
    with_priors = pitch_accuracy(pitch.track_pitch(hard_stem, song.sr, interval_prior=True, key_pcs=key_pcs),
                                 song.notes)
    assert with_priors >= base - 0.03
    assert with_priors > 0.75


def test_beat_prior_reduces_spurious_notes(song, hard_stem):
    _, beats = analysis.estimate_tempo(song.mix, song.sr)
    tr = pitch.track_pitch(hard_stem, song.sr, interval_prior=True)
    ons = nm.detect_onsets(hard_stem, song.sr, beats=beats)
    plain = nm.segment_notes(tr, nm.detect_onsets(hard_stem, song.sr))
    guided = nm.segment_notes(tr, ons, beats=beats)
    def hits(found):
        return sum(1 for r in song.notes if any(abs(n.start - r.start) < 0.08 and n.midi == r.midi for n in found))
    assert hits(guided) >= hits(plain) - 1
    assert len(guided) <= len(plain)
    grid = nm.beat_grid(beats, 12.0)
    assert len(grid) >= 2 * len(beats) - 2


def test_two_pass_tracking_on_enhanced(song, hard_stem):
    sr = song.sr
    tr1 = pitch.track_pitch(hard_stem, sr, interval_prior=True)
    ons1 = nm.detect_onsets(hard_stem, sr)
    nts1 = nm.segment_notes(tr1, ons1)
    out, _ = enhance.enhance_bass(hard_stem, sr, tr1, nts1, ons1, config.Settings())
    tr2 = pitch.track_pitch(out, sr, interval_prior=True)
    a1, a2 = pitch_accuracy(tr1, song.notes), pitch_accuracy(tr2, song.notes)
    assert a2 >= a1 - 0.03


def test_pipeline_auto_with_speed_correction(tmp_path: Path, song):
    inp = tmp_path / "agudo.wav"
    y = _detune(song.mix, song.sr, 30.0)
    save_audio(inp, y, song.sr)
    s = config.Settings(backend="dsp", mode="hybrid", auto=True, pitch_method="salience")
    res = process_file(inp, tmp_path / "out", s, explicit={"pitch_method"})
    assert abs(res.speed_correction_cents - 30.0) < 4.0
    out, sr = load_audio(res.output_path)
    expected = len(y) * 2 ** (res.speed_correction_cents / 1200)
    assert abs(out.shape[-1] - expected) < 0.002 * expected
    assert res.choices["profile"]["valor"] == "vintage"
    assert res.passes["pass2_notes"] > 10
    meta = json.loads(Path(res.notes_json_path).read_text(encoding="utf-8"))
    assert meta["analysis"]["tuning_cents"] > 20
    assert meta["speed_correction_cents"] == res.speed_correction_cents


def test_pipeline_no_auto_no_two_pass(tmp_path: Path, song):
    inp = tmp_path / "m.wav"
    save_audio(inp, song.mix, song.sr)
    s = config.Settings(backend="dsp", mode="enhance", auto=False, two_pass=False, musical_priors=False,
                        speed_correction="off", pitch_method="salience")
    res = process_file(inp, tmp_path / "out", s)
    assert res.speed_correction_cents == 0.0
    assert res.choices == {}
    assert "pass2_notes" not in res.passes
    out, _ = load_audio(res.output_path)
    assert out.shape[-1] == len(song.mix)


def test_crepe_activation_if_available():
    torchcrepe = pytest.importorskip("torchcrepe")
    sr = 16000
    t = np.arange(sr * 3) / sr
    y = (0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 110 * t)
         + 0.2 * np.sin(2 * np.pi * 165 * t)).astype(np.float32)
    cand = 38.0 * 2 ** (np.arange(190) * 20 / 1200)
    times = np.arange(0, 3.0, 0.01)
    act = pitch.crepe_activation(y, sr, cand, times, model="tiny")
    assert act.shape == (len(cand), len(times))
    f = cand[act[:, 100:200].mean(axis=1).argmax()]
    assert abs(1200 * np.log2(f / 55.0)) < 50
    tr = pitch.track_pitch(y, sr, method="fusion", crepe_model="tiny")
    assert np.nanmedian(tr.f0) == pytest.approx(55.0, rel=0.03)
