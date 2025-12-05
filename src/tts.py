from typing import Any

import numpy as np

from .config import settings
from .utils import log


class TTSModel:
    """
    Placeholder TTS model.

    In a real system, this could wrap:
    - Coqui TTS
    - Tacotron + WaveGlow
    - VITS, etc.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.tts_model_name
        log(f"Initialized TTS model: {self.model_name} (placeholder).")

    def synthesize(self, text: str) -> Any:
        """
        Returns dummy audio waveform.
        """
        log("Running TTS (placeholder).")
        # In a real system, return a real waveform at 16kHz
        dummy_audio = np.zeros(16000, dtype="float32")
        sample_rate = 16000
        return dummy_audio, sample_rate

