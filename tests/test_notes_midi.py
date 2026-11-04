from pathlib import Path

from bass_separator import midi, notes as nm, pitch


def _notes(song, sig):
    tr = pitch.track_pitch(sig, song.sr)
    ons = nm.detect_onsets(sig, song.sr)
    return nm.segment_notes(tr, ons)


def test_segmentation_recovers_walking_line(song, hard_stem):
    found = _notes(song, hard_stem)
    hits = 0
    for r in song.notes:
        if any(abs(n.start - r.start) < 0.08 and n.midi == r.midi for n in found):
            hits += 1
    assert hits / len(song.notes) > 0.8
    assert len(found) < 1.5 * len(song.notes)
    assert all(n.end > n.start and 1 <= n.velocity <= 127 for n in found)


def test_midi_roundtrip(tmp_path: Path, song):
    found = _notes(song, song.bass)
    path = tmp_path / "linea.mid"
    midi.write_midi(path, found)
    back = midi.read_midi_notes(path)
    assert len(back) == len(found)
    for (s, e, p, v), n in zip(back, sorted(found, key=lambda n: n.start)):
        assert abs(s - n.start) < 0.005
        assert p == n.midi
        assert v == n.velocity


def test_note_names(song):
    found = _notes(song, song.bass)
    assert all(isinstance(n.name, str) and n.name for n in found)
    d = found[0].to_dict()
    assert {"start", "end", "midi", "f0", "velocity", "name"} <= set(d)
