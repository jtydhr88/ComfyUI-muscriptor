import os
import sys
import time
import uuid

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MUSCRIPTOR_REPO_DIR = os.path.join(CURRENT_DIR, "muscriptor")

# Vendored upstream repo: ComfyUI-muscriptor/muscriptor/muscriptor is the
# actual `muscriptor` package, so the repo root goes on sys.path.
if MUSCRIPTOR_REPO_DIR not in sys.path:
    sys.path.insert(0, MUSCRIPTOR_REPO_DIR)

# muscriptor serializes MIDI through mido, which ComfyUI does not ship.
# Register the bundled stdlib-only writer under the same name rather than
# adding a dependency; a real mido installation wins if present.
try:
    import mido  # noqa: F401
except ImportError:
    from . import midi_shim
    sys.modules["mido"] = midi_shim

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    COMFY_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
    COMFY_MODELS_DIR = os.path.join(COMFY_ROOT, "models")
    COMFY_OUTPUT_DIR = os.path.join(COMFY_ROOT, "output")

MUSCRIPTOR_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "muscriptor")

MODEL_VARIANTS = ["small", "medium", "large"]

_INSTRUMENT_NAMES = (
    "acoustic_piano, electric_piano, chromatic_percussion, organ, "
    "acoustic_guitar, clean_electric_guitar, distorted_electric_guitar, "
    "acoustic_bass, electric_bass, violin, viola, cello, contrabass, "
    "orchestral_harp, timpani, string_ensemble, synth_strings, voice, "
    "orchestra_hit, trumpet, trombone, tuba, french_horn, brass_section, "
    "soprano_and_alto_sax, tenor_sax, baritone_sax, oboe, english_horn, "
    "bassoon, clarinet, flutes, synth_lead, synth_pad, drums"
)


# ============================================================================
# Node: MuScriptor Load Model
# ============================================================================

class MuScriptorLoadModel:
    """Load a MuScriptor audio-to-MIDI transcription model.

    Size keywords (small/medium/large) download from HuggingFace on first
    use and are cached. The weights are license-gated: accept the license on
    the model page and authenticate (`hf auth login` or HF_TOKEN).
    Local .safetensors files in ComfyUI/models/muscriptor/ are also listed
    (put the matching config.json next to the weights).
    """

    @classmethod
    def INPUT_TYPES(s):
        models = list(MODEL_VARIANTS)
        if os.path.isdir(MUSCRIPTOR_MODELS_DIR):
            for f in sorted(os.listdir(MUSCRIPTOR_MODELS_DIR)):
                if f.endswith(".safetensors"):
                    models.append(f)
        return {
            "required": {
                "model": (models, {"default": "medium", "tooltip":
                    "small (103M) / medium (307M) / large (1.4B) download from "
                    "HuggingFace on first use; .safetensors entries are local "
                    "files from ComfyUI/models/muscriptor/"}),
            },
            "optional": {
                "dtype": (["auto", "float32", "float16", "bfloat16"],
                    {"default": "auto", "tooltip":
                     "auto = float32 (float16 on MPS); CUDA uses fp16 autocast "
                     "internally either way"}),
            },
        }

    RETURN_TYPES = ("MUSCRIPTOR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MuScriptor"

    def load(self, model, dtype="auto"):
        import comfy.model_management as model_management
        from muscriptor import TranscriptionModel
        from muscriptor.utils.download import ModelDownloadError

        source = model
        if model not in MODEL_VARIANTS:
            source = os.path.join(MUSCRIPTOR_MODELS_DIR, model)
            if not os.path.exists(source):
                raise FileNotFoundError(f"Checkpoint not found: {source}")

        device = model_management.get_torch_device()
        print(f"[MuScriptor] Loading model '{model}' on {device}...")
        try:
            tm = TranscriptionModel.load_model(
                source,
                device=device,
                dtype=None if dtype == "auto" else dtype,
            )
        except ModelDownloadError as e:
            raise RuntimeError(
                f"MuScriptor weights download failed: {e}\n"
                "The weights are gated on HuggingFace: accept the license at "
                f"https://huggingface.co/MuScriptor/muscriptor-{model} and "
                "authenticate (`hf auth login` or set HF_TOKEN)."
            ) from e
        print("[MuScriptor] Model loaded")
        return (tm,)


# ============================================================================
# Node: MuScriptor Transcribe
# ============================================================================

class MuScriptorTranscribe:
    """Transcribe AUDIO into a MIDI file (bytes).

    Multi-channel audio is mixed down to mono and resampled to the model's
    16 kHz internally. Note: the model recovers onset/offset timing, pitch
    and instrument — not velocity.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MUSCRIPTOR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "instruments": ("STRING", {"default": "", "tooltip":
                    "Comma-separated instrument groups to restrict decoding to; "
                    "empty = detect automatically. Case-insensitive, unambiguous "
                    "abbreviations work (e.g. 'timp,dist'). Names: "
                    + _INSTRUMENT_NAMES}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip":
                    "Temperature sampling instead of greedy decoding"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.05,
                    "max": 2.0, "step": 0.05, "tooltip":
                    "Only used when use_sampling is on"}),
                "beam_size": ("INT", {"default": 1, "min": 1, "max": 8,
                    "tooltip": ">= 2 enables beam search: slower, can be more "
                    "accurate"}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 16,
                    "tooltip": "5-second chunks per forward pass. 0 = auto. "
                    "> 1 requires prelude_forcing off and trades chunk-boundary "
                    "quality for throughput"}),
                "prelude_forcing": ("BOOLEAN", {"default": True, "tooltip":
                    "Teacher-force notes sustained across chunk boundaries into "
                    "the next chunk (higher quality, forces batch_size 1)"}),
            },
        }

    RETURN_TYPES = ("MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "transcribe"
    CATEGORY = "MuScriptor"

    def transcribe(self, model, audio, instruments="", use_sampling=False,
                   temperature=1.0, beam_size=1, batch_size=0,
                   prelude_forcing=True):
        import comfy.model_management as model_management
        import comfy.utils
        from muscriptor.events import NoteStartEvent, ProgressEvent
        from muscriptor.tokenizer.mt3 import resolve_instrument_names

        names = None
        tokens = [t.strip() for t in instruments.split(",") if t.strip()]
        if tokens:
            names = resolve_instrument_names(tokens)
            print(f"[MuScriptor] Restricting to instruments: {', '.join(names)}")

        waveform = audio["waveform"][0]  # [C, T]; mono mixdown happens inside
        sample_rate = audio["sample_rate"]

        events = model.transcribe(
            (waveform, sample_rate),
            use_sampling=use_sampling,
            temperature=temperature,
            instruments=names,
            batch_size=batch_size if batch_size > 0 else None,
            beam_size=beam_size,
            prelude_forcing=prelude_forcing,
        )

        pbar = comfy.utils.ProgressBar(1)
        note_count = [0]

        def tracked():
            for ev in events:
                if isinstance(ev, ProgressEvent):
                    model_management.throw_exception_if_processing_interrupted()
                    pbar.update_absolute(ev.completed, ev.total)
                elif isinstance(ev, NoteStartEvent):
                    note_count[0] += 1
                yield ev

        midi_bytes = model.events_to_midi_bytes(tracked())
        print(f"[MuScriptor] Transcribed {note_count[0]} notes "
              f"({len(midi_bytes)} bytes of MIDI)")
        return (midi_bytes,)


# ============================================================================
# Node: MuScriptor Save MIDI
# ============================================================================

class MuScriptorSaveMIDI:
    """Write MIDI bytes to a .mid file in the ComfyUI output directory."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "midi": ("MIDI",),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "midi"}),
                "filename_prefix": ("STRING", {"default": "muscriptor"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "MuScriptor"
    OUTPUT_NODE = True

    def save(self, midi, output_dir="midi", filename_prefix="muscriptor"):
        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        save_path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}.mid")
        with open(save_path, "wb") as f:
            f.write(midi)

        print(f"[MuScriptor] Saved: {save_path}")
        rel_path = os.path.relpath(save_path, COMFY_OUTPUT_DIR).replace("\\", "/")
        # Report the file as a ui output so /history keeps it and /view serves it.
        subfolder, filename = os.path.split(rel_path)
        return {
            "ui": {"midi": [{"filename": filename, "subfolder": subfolder,
                             "type": "output"}]},
            "result": (rel_path,),
        }


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "MuScriptorLoadModel": MuScriptorLoadModel,
    "MuScriptorTranscribe": MuScriptorTranscribe,
    "MuScriptorSaveMIDI": MuScriptorSaveMIDI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuScriptorLoadModel": "MuScriptor Load Model",
    "MuScriptorTranscribe": "MuScriptor Transcribe",
    "MuScriptorSaveMIDI": "MuScriptor Save MIDI",
}
