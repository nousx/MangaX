import os
import sys

# Use an absolute path, to avoid a KeyError on importlib.invalidate_caches()
_current_dir = os.path.dirname(os.path.abspath(__file__))
_inpainting_dir = os.path.dirname(_current_dir)
if _inpainting_dir not in sys.path:
    sys.path.append(_inpainting_dir)
