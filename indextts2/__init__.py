"""
IndexTTS2 integration package for ComfyUI
This package provides model loading and inference wrappers for IndexTTS2.
"""

from .model_loader import IndexTTS2Loader
from .infer import IndexTTS2Engine

__all__ = ["IndexTTS2Loader", "IndexTTS2Engine"]
