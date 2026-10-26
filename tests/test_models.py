import asyncio
from alignment_harness.models.clients import MockModelClient
from alignment_harness.models.base import GenerationRequest

def test_mock_client():
    async def _run():
        client = MockModelClient("test-model", "A")
        res = await client.generate(GenerationRequest(prompt="Hello"))
        assert res.text == "A"
        assert res.latency_ms > 0
    asyncio.run(_run())
