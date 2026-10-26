import re
from typing import List
from alignment_harness.benchmarks.base import BaseBenchmark, BenchmarkSample

class MMLUBenchmark(BaseBenchmark):
    def __init__(self, concurrency_limit: int = 16):
        super().__init__(name="MMLU", concurrency_limit=concurrency_limit)

    def load_dataset(self) -> List[BenchmarkSample]:
        return [
            BenchmarkSample(sample_id="mmlu_1", prompt="What is BST search complexity?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)\nAnswer:", expected_answer="B"),
            BenchmarkSample(sample_id="mmlu_2", prompt="What drops units in NN?\nA) L1\nB) BatchNorm\nC) Dropout\nD) Weight Decay\nAnswer:", expected_answer="C"),
            BenchmarkSample(sample_id="mmlu_3", prompt="What is the objective in DPO?\nA) Policy Gradient\nB) Implicit Reward Optimization\nC) Heuristic Search\nD) Random Walk\nAnswer:", expected_answer="B"),
            BenchmarkSample(sample_id="mmlu_4", prompt="What measures inter-annotator agreement?\nA) Perplexity\nB) Cohen's Kappa\nC) BLEU\nD) Latency\nAnswer:", expected_answer="B")
        ]

    def evaluate_sample(self, sample: BenchmarkSample, model_output: str) -> bool:
        match = re.search(r"\b([A-D])\b", model_output.strip().upper())
        if match:
            return match.group(1) == sample.expected_answer
        return sample.expected_answer.lower() in model_output.strip().lower()
