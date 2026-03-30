"""Tests for Phase 4 contradiction candidate selection and LLM parsing."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

from app.models.document import Chunk, ChunkPosition
from app.services.passes.contradiction import ContradictionDetectionPass
from app.services.similarity import SimilarityService


def test_similarity_service_finds_only_candidate_pairs_in_requested_range() -> None:
    """Candidate pair selection should keep only the requested similarity band."""
    new_chunks = [
        _chunk("a", "Policy states that remote work is allowed.", "policy-a.txt", [1.0, 0.0, 0.0]),
        _chunk("b", "Policy states that remote work is limited.", "policy-b.txt", [0.8, 0.6, 0.0]),
        _chunk("c", "An unrelated benefits note.", "benefits.txt", [0.0, 0.0, 1.0]),
    ]
    existing_chunks = [
        _chunk(
            "d",
            "The indexed policy discusses remote work timing.",
            "indexed-policy.txt",
            [0.75, -math.sqrt(1 - 0.75**2), 0.0],
        ),
        _chunk("e", "A near-identical indexed policy.", "indexed-copy.txt", [0.96, -0.28, 0.0]),
    ]

    pairs = SimilarityService().find_pairs_in_range(
        new_chunks,
        existing_chunks,
        min_similarity=0.7,
        max_similarity=0.95,
    )

    pair_keys = {
        (pair.left_chunk.id, pair.right_chunk.id, pair.comparison_scope, pair.severity)
        for pair in pairs
    }

    assert pair_keys == {
        ("a", "b", "new_vs_new", "candidate"),
        ("a", "d", "new_vs_existing", "candidate"),
    }


def test_contradiction_pass_parses_valid_json_and_treats_malformed_content_as_consistent() -> None:
    """The parser should normalize valid JSON and ignore malformed model output."""
    contradiction_pass = ContradictionDetectionPass()

    parsed = contradiction_pass.parse_llm_response(
        """```json
{"status":"CONTRADICTION","claim_a":"A","claim_b":"B","explanation":"Direct conflict.","severity":"HIGH"}
```"""
    )

    assert parsed == {
        "status": "CONTRADICTION",
        "claim_a": "A",
        "claim_b": "B",
        "explanation": "Direct conflict.",
        "severity": "high",
    }
    assert contradiction_pass.parse_llm_response("not-json") == {"status": "CONSISTENT"}


def test_contradiction_prompt_requires_a_direct_same_fact_conflict() -> None:
    """The contradiction prompt should distinguish true conflicts from different subtopics."""
    prompt = ContradictionDetectionPass.SYSTEM_PROMPT

    assert "Follow these steps strictly" in prompt
    assert "EXACT same thing" in prompt
    assert "Passage A discusses hotel reimbursement, Passage B discusses meal allowances" in prompt
    assert "When in doubt, mark as CONSISTENT" in prompt


def test_contradiction_pass_includes_same_document_pairs_by_default(monkeypatch) -> None:
    """Same-document pairs should remain eligible unless explicitly disabled."""
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"status":"CONSISTENT"}'))],
                usage=SimpleNamespace(prompt_tokens=40, completion_tokens=10),
            )

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    pair = SimilarityService().find_pairs_in_range(
        [
            _chunk(
                "a",
                "Hotel reimbursement is capped at $125 per night.",
                "travel-policy-old.md",
                [1.0, 0.0, 0.0],
                document_id="doc-travel",
            ),
            _chunk(
                "b",
                "Meal allowances are capped at $40 per day.",
                "travel-policy-old.md",
                [0.8, 0.6, 0.0],
                document_id="doc-travel",
            ),
        ],
        min_similarity=0.7,
        max_similarity=0.95,
    )[0]

    findings, stats = _run(ContradictionDetectionPass().run([pair], api_key="sk-test"))

    assert findings == []
    assert stats.candidate_pairs_considered == 1
    assert stats.llm_calls_made == 1
    assert stats.failed_calls == 0


def test_contradiction_pass_can_skip_same_document_pairs_when_requested() -> None:
    """The opt-out flag should filter same-document pairs before any LLM call."""
    pair = SimilarityService().find_pairs_in_range(
        [
            _chunk(
                "a",
                "Hotel reimbursement is capped at $125 per night.",
                "travel-policy-old.md",
                [1.0, 0.0, 0.0],
                document_id="doc-travel",
            ),
            _chunk(
                "b",
                "Meal allowances are capped at $40 per day.",
                "travel-policy-old.md",
                [0.8, 0.6, 0.0],
                document_id="doc-travel",
            ),
        ],
        min_similarity=0.7,
        max_similarity=0.95,
    )[0]

    findings, stats = _run(
        ContradictionDetectionPass(include_same_document_pairs=False).run(
            [pair],
            api_key="sk-test",
        )
    )

    assert findings == []
    assert stats.candidate_pairs_considered == 0
    assert stats.llm_calls_made == 0
    assert stats.failed_calls == 0


def test_contradiction_pass_run_builds_findings_and_usage_stats(monkeypatch) -> None:
    """The contradiction pass should parse a mocked OpenAI response into findings and cost stats."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"status":"CONTRADICTION","claim_a":"Passage A says shipping is free.",'
                                '"claim_b":"Passage B says shipping costs $10.","explanation":"The pricing '
                                'claims cannot both be true.","severity":"HIGH"}'
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40),
            )

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    pair = SimilarityService().find_pairs_in_range(
        [
            _chunk("a", "Shipping is free for all orders.", "pricing-a.txt", [1.0, 0.0, 0.0]),
            _chunk("b", "Shipping costs $10 for every order.", "pricing-b.txt", [0.8, 0.6, 0.0]),
        ],
        min_similarity=0.7,
        max_similarity=0.95,
    )[0]

    findings, stats = _run(
        ContradictionDetectionPass().run(
            [pair],
            api_key="sk-test",
        )
    )

    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].claim_a == "Passage A says shipping is free."
    assert findings[0].claim_b == "Passage B says shipping costs $10."
    assert findings[0].chunks_involved[0].source_filename == "pricing-a.txt"
    assert findings[0].chunks_involved[1].source_filename == "pricing-b.txt"
    assert stats.llm_calls_made == 1
    assert stats.failed_calls == 0
    assert stats.candidate_pairs_considered == 1
    assert stats.prompt_tokens == 120
    assert stats.completion_tokens == 40
    assert stats.estimated_cost_usd == 0.000084


def _chunk(
    chunk_id: str,
    text: str,
    filename: str,
    embedding: list[float],
    *,
    document_id: str | None = None,
) -> Chunk:
    """Build a chunk for contradiction and similarity tests."""
    return Chunk(
        id=chunk_id,
        text=text,
        parent_document_id=document_id or f"doc-{chunk_id}",
        parent_document_name=filename,
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=len(text)),
        metadata={},
        embedding=embedding,
    )


def _run(awaitable):
    """Run a single coroutine without introducing a pytest async dependency."""
    import asyncio

    return asyncio.run(awaitable)
