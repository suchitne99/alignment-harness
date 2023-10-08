import re
from typing import List
from alignment_harness.benchmarks.base import BaseBenchmark, BenchmarkSample

class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, concurrency_limit: int = 16):
        super().__init__(name="GSM8K", concurrency_limit=concurrency_limit)

    def load_dataset(self) -> List[BenchmarkSample]:
        return [
            BenchmarkSample(sample_id="gsm_1", prompt="Natalia sold 48 clips in April, half as many in May. Total?\nSolution:", expected_answer="72")
        ]

    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        numbers = re.findall(r"[-+]?\d*\.?\d+", model_output.replace(",", ""))
        return numbers and (numbers[-1] == sample.expected_answer or sample.expected_answer in numbers)
