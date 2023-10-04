import time
import asyncio
from typing import Optional
from alignment_harness.models.base import BaseModelClient, GenerationRequest, GenerationResponse

class MockModelClient(BaseModelClient):
    def __init__(self, model_name: str, default_response: str = "B"):
        super().__init__(model_name)
        self.default_response = default_response

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        await asyncio.sleep(0.01)
        return GenerationResponse(
            text=self.default_response,
            model_name=self.model_name,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(self.default_response.split()),
            latency_ms=12.5
        )

# Backoff jitter
