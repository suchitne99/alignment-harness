import asyncio
from pydantic import BaseModel
from alignment_harness.models.base import BaseModelClient, GenerationRequest

class PairwiseJudgment(BaseModel):
    winner: str
    confidence: float
    position_consistent: bool

class JudgeEvaluator:
    def __init__(self, judge_client: BaseModelClient):
        self.judge = judge_client

    async def evaluate_pair(self, prompt: str, resp_a: str, resp_b: str) -> PairwiseJudgment:
        p1 = f"[Prompt]\n{prompt}\n[Assistant 1]\n{resp_a}\n[Assistant 2]\n{resp_b}\n[Decision]"
        p2 = f"[Prompt]\n{prompt}\n[Assistant 1]\n{resp_b}\n[Assistant 2]\n{resp_a}\n[Decision]"
        r1, r2 = await asyncio.gather(
            self.judge.generate(GenerationRequest(prompt=p1)),
            self.judge.generate(GenerationRequest(prompt=p2))
        )
        w1 = "model_a" if "1" in r1.text else "model_b"
        w2 = "model_b" if "1" in r2.text else "model_a"
        if w1 == w2:
            return PairwiseJudgment(winner=w1, confidence=0.95, position_consistent=True)
        return PairwiseJudgment(winner="tie", confidence=0.50, position_consistent=False)
