"""Minimal stand-in for the parts of `mido` that muscriptor uses.

ComfyUI does not ship mido and this plugin adds no Python dependencies, so
when `import mido` fails, nodes.py registers this module under
sys.modules["mido"] before the vendored muscriptor package is imported.
A real mido installation, if present, is used instead.

Covered surface (see muscriptor/tokenizer/notes.py and utils/midi.py):
    Message("note_on" | "note_off" | "program_change", ...)
    MetaMessage("set_tempo" | "track_name", ...)
    MidiTrack()                        — a list of messages
    MidiFile(type=1, ticks_per_beat=)  — .tracks, .save(filename)/.save(file=)
    second2tick(second, ticks_per_beat, tempo)

Messages carry delta times in ticks (the `time` attribute), matching mido's
track semantics. Serialization is a plain type-1 Standard MIDI File without
running status.
"""

import struct


def second2tick(second, ticks_per_beat, tempo):
    # tempo is microseconds per quarter note
    scale = tempo * 1e-6 / ticks_per_beat
    return int(round(second / scale))


def _encode_varlen(value):
    if value < 0:
        raise ValueError(f"negative delta time: {value}")
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(out)


class Message:
    is_meta = False

    def __init__(self, type, note=0, velocity=64, program=0, channel=0, time=0):
        if type not in ("note_on", "note_off", "program_change"):
            raise ValueError(f"unsupported message type: {type!r}")
        self.type = type
        self.note = note
        self.velocity = velocity
        self.program = program
        self.channel = channel
        self.time = time

    def _payload(self):
        if self.type == "note_on":
            return bytes([0x90 | self.channel, self.note, self.velocity])
        if self.type == "note_off":
            return bytes([0x80 | self.channel, self.note, self.velocity])
        return bytes([0xC0 | self.channel, self.program])

    def __repr__(self):
        return f"Message({self.type!r}, channel={self.channel}, time={self.time})"


class MetaMessage:
    is_meta = True

    def __init__(self, type, tempo=500000, name="", time=0):
        if type not in ("set_tempo", "track_name", "end_of_track"):
            raise ValueError(f"unsupported meta message type: {type!r}")
        self.type = type
        self.tempo = tempo
        self.name = name
        self.time = time

    def _payload(self):
        if self.type == "set_tempo":
            return b"\xff\x51\x03" + self.tempo.to_bytes(3, "big")
        if self.type == "track_name":
            data = self.name.encode("utf-8")
            return b"\xff\x03" + _encode_varlen(len(data)) + data
        return b"\xff\x2f\x00"

    def __repr__(self):
        return f"MetaMessage({self.type!r}, time={self.time})"


class MidiTrack(list):
    pass


class MidiFile:
    def __init__(self, type=1, ticks_per_beat=480):
        self.type = type
        self.ticks_per_beat = ticks_per_beat
        self.tracks = []

    def _serialize_track(self, track):
        data = bytearray()
        for msg in track:
            data += _encode_varlen(int(msg.time))
            data += msg._payload()
        if not (track and getattr(track[-1], "type", None) == "end_of_track"):
            data += b"\x00\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)

    def save(self, filename=None, file=None):
        data = b"MThd" + struct.pack(
            ">IHHH", 6, self.type, len(self.tracks), self.ticks_per_beat
        )
        for track in self.tracks:
            data += self._serialize_track(track)
        if file is not None:
            file.write(data)
        elif filename is not None:
            with open(filename, "wb") as f:
                f.write(data)
        else:
            raise ValueError("save() needs a filename or a file object")
