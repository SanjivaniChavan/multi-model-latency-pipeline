from typing import Any

import numpy as np

from .utils import log


class SimpleVAD:
    """
    Very simple placeholder VAD.

    In a real system, this could wrap WebRTC VAD, Silero VAD, etc.
    For now, we just 'pretend' the audio is speech and pass it through.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def run(self, audio: np.ndarray, sample_rate: int) -> Any:
        """
        audio: np.ndarray of shape (T,) or (1, T)
        Returns a possibly-trimmed audio.
        """
        log("Running VAD (placeholder: no trimming applied).")
        return audio  # placeholder - no trimming yet
