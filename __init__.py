"""
ComfyUI-muscriptor

Audio-to-MIDI multi-instrument transcription (Kyutai MuScriptor).
"""

import os

# Track initialization status
INIT_SUCCESS = False

if not os.environ.get('PYTEST_CURRENT_TEST'):
    print("[ComfyUI-muscriptor] Initializing custom node...")

    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[ComfyUI-muscriptor] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        import traceback
        print(f"[ComfyUI-muscriptor] [WARNING] Failed to import node classes: {e}")
        print(f"[ComfyUI-muscriptor] Traceback:\n{traceback.format_exc()}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    if INIT_SUCCESS:
        print("[ComfyUI-muscriptor] [OK] Loaded successfully!")
    else:
        print("[ComfyUI-muscriptor] [ERROR] Failed to load - check errors above")

else:
    print("[ComfyUI-muscriptor] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.1.0"
