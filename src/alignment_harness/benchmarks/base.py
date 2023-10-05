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
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    @abstractmethod
    def load_dataset(self) -> List[BenchmarkSample]:
        pass

    @abstractmethod
    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        pass

# Semaphore optimization

# uvloop hook
