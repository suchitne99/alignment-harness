import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from alignment_harness.models.base import BaseModelClient, GenerationRequest

class BenchmarkSample(BaseModel):
    sample_id: str
    prompt: str
    expected_answer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkResult(BaseModel):
    benchmark_name: str
    model_name: str
    total_samples: int
    correct_samples: int
    accuracy: float
    average_latency_ms: float

class BaseBenchmark(ABC):
    def __init__(self, name: str, concurrency_limit: int = 16):
        self.name = name
        self.concurrency_limit = concurrency_limit

    @abstractmethod
    def load_dataset(self) -> List[BenchmarkSample]:
        pass

    @abstractmethod
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        pass

    async def run(self, model_client: BaseModelClient) -> BenchmarkResult:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        samples = self.load_dataset()
        correct = 0
        total_latency = 0.0

        async def _eval_one(sample: BenchmarkSample):
            nonlocal correct, total_latency
            async with semaphore:
                req = GenerationRequest(prompt=sample.prompt)
                res = await model_client.generate(req)
                total_latency += res.latency_ms
                if self.evaluate_sample(sample, res.text):
                    correct += 1

        await asyncio.gather(*(_eval_one(s) for s in samples))
        total = len(samples)
        acc = correct / total if total > 0 else 0.0
        avg_lat = total_latency / total if total > 0 else 0.0
        return BenchmarkResult(
            benchmark_name=self.name,
            model_name=model_client.model_name,
            total_samples=total,
            correct_samples=correct,
            accuracy=acc,
            average_latency_ms=avg_lat
        )
