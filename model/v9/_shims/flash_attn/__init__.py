"""See flash_attn_interface.py. This package exists ONLY so XPert's model_utils.py can be imported on a
machine without CUDA; it is never used when the real flash_attn is installed."""
from . import flash_attn_interface          # noqa: F401
