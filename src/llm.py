from typing import List

from .config import settings
from .utils import log


class LLMClient:
    """
    Placeholder LLM client.

    In a real version, this could call:
    - OpenAI API
    - Local LLaMA/Mistral via text-generation-inference
    - vLLM, etc.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.llm_model_name
        log(f"Initialized LLM client with model: {self.model_name} (placeholder).")

    def generate(self, prompt: str, context_chunks: List[str]) -> str:
        """
        Combine prompt + retrieved context and return a response.
        Currently returns a dummy response.
        """
        log("Running LLM generate (placeholder).")
        joined_context = "\n\n".join(context_chunks)
        return (
            f"[DUMMY LLM RESPONSE]\n"
            f"Prompt: {prompt}\n\n"
            f"Context used:\n{joined_context[:500]}"
        )
