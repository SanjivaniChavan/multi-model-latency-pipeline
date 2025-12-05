from typing import Tuple

import numpy as np

from .utils import log


class ASRModel:
    """
    Placeholder ASR model interface.

    For a real system, this could wrap OpenAI Whisper, Vosk, Nemo ASR, etc.
    Here we provide a simple stub so the pipeline structure is clear.
    """

    def __init__(self, model_name: str = "whisper-small"):
        self.model_name = model_name
        log(f"Initialized ASR model: {model_name} (placeholder).")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        Returns: (transcript, confidence_score).
        Currently returns a dummy transcript.
        """
        log("Running ASR (placeholder).")
        # In a real model, we'd call the ASR here.
        transcript = "this is a dummy transcription from asr placeholder"
        confidence = 0.95
        return transcript, confidence
