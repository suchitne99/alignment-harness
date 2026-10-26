import asyncio
from alignment_harness.models.clients import MockModelClient
from alignment_harness.benchmarks.mmlu import MMLUBenchmark
from alignment_harness.benchmarks.gsm8k import GSM8KBenchmark
from alignment_harness.evaluators.judge import JudgeEvaluator
from alignment_harness.evaluators.elo import EloRatingEngine

def test_mmlu_benchmark_execution():
    async def _run():
        client = MockModelClient("test-model", default_response="Option B")
        bench = MMLUBenchmark()
        result = await bench.run(client)
        assert result.total_samples == 4
        assert result.accuracy > 0.0
    asyncio.run(_run())

def test_gsm8k_numerical_extraction():
    async def _run():
        client = MockModelClient("test-model", default_response="The final answer is 72.")
        bench = GSM8KBenchmark()
        result = await bench.run(client)
        assert result.correct_samples >= 1
    asyncio.run(_run())

def test_judge_position_swap_consistency():
    async def _run():
        judge = MockModelClient("judge-gpt4", default_response="[[A]] Assistant 1 is superior.")
        evaluator = JudgeEvaluator(judge)
        judgment = await evaluator.evaluate_pair("Prompt", "Response A", "Response B")
        assert judgment.position_consistent is False
        assert judgment.winner == "tie"
    asyncio.run(_run())

def test_elo_calculation():
    engine = EloRatingEngine(k_factor=32.0, base_elo=1000.0)
    engine.record_match("model_a", "model_b", "model_a")
    leaderboard = engine.get_leaderboard()
    assert leaderboard[0].model_name == "model_a"
    assert leaderboard[0].elo > 1000.0
    assert leaderboard[1].elo < 1000.0
