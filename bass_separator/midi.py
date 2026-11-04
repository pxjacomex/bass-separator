"""Escritor mínimo de archivos MIDI estándar (formato 0), sin dependencias."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def _varlen(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi(path: str | Path, notes: Iterable, tempo_bpm: float = 120.0, ppq: int = 480,
               program: int = 32, channel: int = 0) -> None:
    """Escribe ``notes`` (objetos con ``start``, ``end``, ``midi``, ``velocity``) como SMF tipo 0.

    ``program`` 32 = *Acoustic Bass* en General MIDI.
    """
    ticks_per_s = ppq * tempo_bpm / 60.0
    events = []  # (tick, orden, bytes)
    for n in notes:
        on = int(round(n.start * ticks_per_s))
        off = max(on + 1, int(round(n.end * ticks_per_s)))
        vel = int(max(1, min(127, n.velocity)))
        pitch = int(max(0, min(127, n.midi)))
        events.append((off, 0, bytes([0x80 | channel, pitch, 0])))
        events.append((on, 1, bytes([0x90 | channel, pitch, vel])))
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    us_per_beat = int(round(60_000_000 / tempo_bpm))
    track += _varlen(0) + bytes([0xFF, 0x51, 0x03]) + us_per_beat.to_bytes(3, "big")
    track += _varlen(0) + bytes([0xFF, 0x03]) + _varlen(len(b"contrabajo")) + b"contrabajo"
    track += _varlen(0) + bytes([0xC0 | channel, program])
    last = 0
    for tick, _, data in events:
        track += _varlen(tick - last) + data
        last = tick
    track += _varlen(0) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ppq.to_bytes(2, "big")
    body = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header + body)


def read_midi_notes(path: str | Path) -> list[tuple[float, float, int, int]]:
    """Lector sencillo (solo para pruebas): devuelve (inicio_s, fin_s, nota, velocidad)."""
    data = Path(path).read_bytes()
    assert data[:4] == b"MThd"
    ppq = int.from_bytes(data[12:14], "big")
    pos = 14
    assert data[pos:pos + 4] == b"MTrk"
    length = int.from_bytes(data[pos + 4:pos + 8], "big")
    pos += 8
    end = pos + length
    tick = 0
    tempo = 500000
    active: dict[int, tuple[int, int]] = {}
    notes = []
    status = 0

    def read_var(p):
        v = 0
        while True:
            b = data[p]
            p += 1
            v = (v << 7) | (b & 0x7F)
            if not b & 0x80:
                return v, p

    while pos < end:
        delta, pos = read_var(pos)
        tick += delta
        b = data[pos]
        if b == 0xFF:
            meta = data[pos + 1]
            ln, p2 = read_var(pos + 2)
            if meta == 0x51:
                tempo = int.from_bytes(data[p2:p2 + 3], "big")
            pos = p2 + ln
            continue
        if b & 0x80:
            status = b
            pos += 1
        kind = status & 0xF0
        if kind in (0x80, 0x90):
            pitch, vel = data[pos], data[pos + 1]
            pos += 2
            if kind == 0x90 and vel > 0:
                active[pitch] = (tick, vel)
            elif pitch in active:
                t0, v0 = active.pop(pitch)
                s = t0 * tempo / 1e6 / ppq
                e = tick * tempo / 1e6 / ppq
                notes.append((s, e, pitch, v0))
        elif kind in (0xC0, 0xD0):
            pos += 1
        else:
            pos += 2
    return sorted(notes)
