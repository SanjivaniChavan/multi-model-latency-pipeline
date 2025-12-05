from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the multi-model pipeline.
    This can later be extended to load from env vars / .env file.
    """

    # Device config
    device: str = "cuda"  # or "cpu"

    # Model names (can be swapped easily)
    asr_model_name: str = "openai/whisper-small"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "gpt-3.5-turbo"  # or local LLaMA via API
    tts_model_name: str = "basic-tts"

    # FAISS index paths
    faiss_index_path: str = "data/rag.index"
    faiss_metadata_path: str = "data/rag_metadata.json"

    # RAG settings
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
