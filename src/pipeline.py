from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .asr import ASRModel
from .config import settings
from .embeddings import embed_texts
from .llm import LLMClient
from .rag_faiss import FaissRAGRetriever
from .tts import TTSModel
from .utils import log, timed
from .vad import SimpleVAD


app = FastAPI(title="Multi-Model Latency Pipeline")

# Instantiate components (in real systems, you'd handle warmup, GPU placement, etc.)
vad = SimpleVAD()
asr_model = ASRModel(settings.asr_model_name)
rag = FaissRAGRetriever()
llm = LLMClient()
tts_model = TTSModel()


class ChatRequest(BaseModel):
    # For simplicity, we accept text as "simulated ASR"
    text: str


class ChatResponse(BaseModel):
    transcript: str
    retrieved_context: List[str]
    llm_response: str
    timings_ms: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    timings = {}

    # 1) "VAD + ASR" (simulated from text)
    with timed("ASR") as t_asr:
        transcript = req.text  # in a real version, you'd pass audio -> ASR
        confidence = 0.99
    timings["asr_ms"] = t_asr["elapsed_ms"]

    # 2) Embeddings + RAG
    with timed("Embeddings + RAG") as t_rag:
        # We'll just simulate some text chunks as retrieved context
        # In a real system, you'd embed transcript, then query FAISS.
        retrieved = [
            "This is a dummy knowledge chunk 1.",
            "This is a dummy knowledge chunk 2.",
        ]
    timings["rag_ms"] = t_rag["elapsed_ms"]

    # 3) LLM
    with timed("LLM") as t_llm:
        llm_response = llm.generate(prompt=transcript, context_chunks=retrieved)
    timings["llm_ms"] = t_llm["elapsed_ms"]

    # 4) TTS
    with timed("TTS") as t_tts:
        audio, sr = tts_model.synthesize(llm_response)
    timings["tts_ms"] = t_tts["elapsed_ms"]

    total_ms = (
        timings["asr_ms"]
        + timings["rag_ms"]
        + timings["llm_ms"]
        + timings["tts_ms"]
    )
    timings["total_ms"] = total_ms
    log(f"Total pipeline latency: {total_ms:.2f} ms")

    return ChatResponse(
        transcript=transcript,
        retrieved_context=retrieved,
        llm_response=llm_response,
        timings_ms=timings,
    )
