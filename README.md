# ComfyUI-muscriptor

ComfyUI custom nodes for [MuScriptor](https://github.com/muscriptor/muscriptor),
Kyutai's multi-instrument audio-to-MIDI transcription model
([paper](https://arxiv.org/abs/2607.08168)).

Zero extra Python dependencies: everything the transcription path needs
(torch, numpy, einops, safetensors, huggingface_hub) already ships with
ComfyUI. The one upstream dependency ComfyUI lacks — `mido`, used only to
serialize MIDI files — is replaced by a bundled stdlib-only writer
(`midi_shim.py`); if `mido` happens to be installed it is used instead.

## Nodes

| Node | In | Out | Notes |
|---|---|---|---|
| **MuScriptor Load Model** | — | `MUSCRIPTOR_MODEL` | small (103M) / medium (307M) / large (1.4B), or a local `.safetensors` from `ComfyUI/models/muscriptor/` |
| **MuScriptor Transcribe** | `MUSCRIPTOR_MODEL`, `AUDIO` | `MIDI` (bytes) | optional instrument restriction, sampling / beam search, batching |
| **MuScriptor Save MIDI** | `MIDI` | `STRING` (path) | writes `output/midi/<prefix>_<ts>_<id>.mid` and reports it as a `ui.midi` output (`/history`, `/view`) |

Typical chain: `LoadAudio → MuScriptor Transcribe → MuScriptor Save MIDI`
(with `MuScriptor Load Model` feeding the transcriber).

## Model weights (HuggingFace login required)

The weights are hosted on [HuggingFace](https://huggingface.co/MuScriptor)
gated behind a CC BY-NC 4.0 license:

1. Accept the license on the model page, e.g.
   [muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium)
   (access is granted automatically).
2. Authenticate on the machine running ComfyUI: `hf auth login`, or set the
   `HF_TOKEN` environment variable.

Weights download on first use and are cached. Alternatively drop a
`.safetensors` checkpoint (with its `config.json` next to it) into
`ComfyUI/models/muscriptor/` and pick it in the loader.

## Notes

- Multi-channel input is mixed down to mono and resampled to 16 kHz
  internally; any `AUDIO`-producing node works as a source.
- The model recovers onset/offset timing, pitch, and instrument — **not
  velocity** (all notes are written at a fixed velocity).
- `instruments` on the Transcribe node is a hard constraint: listed groups
  are the only ones that can appear in the output. Empty = auto-detect.
- The vendored upstream repo lives under `muscriptor/`; only its Python
  package is used (no server / web UI / CLI).

## License

Node code: MIT. Upstream muscriptor code: MIT. Model weights: CC BY-NC 4.0
(non-commercial).
