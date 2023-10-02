from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 512
    stop_sequences: List[str] = Field(default_factory=list)

class GenerationResponse(BaseModel):
    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

class BaseModelClient(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        pass

# Reasoning token filter
