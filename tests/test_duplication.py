"""Tests for Phase 2 similarity and duplication detection."""

from __future__ import annotations

import math

from app.models.document import Chunk, ChunkPosition
from app.services.passes.duplication import DuplicationDetectionPass
from app.services.similarity import SimilarityPair, SimilarityService


def test_similarity_service_finds_exact_near_and_brownfield_matches() -> None:
    """Cosine similarity should surface exact, near, and new-vs-existing matches."""
    new_chunks = [
        _chunk("a", "alpha", "alpha.txt", [1.0, 0.0]),
        _chunk("b", "alpha copy", "alpha-copy.txt", [1.0, 0.0]),
        _chunk("c", "alpha paraphrase", "alpha-paraphrase.txt", [0.9, math.sqrt(1 - 0.9**2)]),
        _chunk("d", "unrelated", "other.txt", [0.0, 1.0]),
    ]
    existing_chunks = [
        _chunk("e", "alpha existing", "indexed.txt", [1.0, 0.0]),
    ]

    pairs = SimilarityService(
        exact_threshold=0.98,
        near_duplicate_threshold=0.85,
    ).find_similar_chunks(new_chunks, existing_chunks)

    exact_pairs = [
        pair
        for pair in pairs
        if pair.comparison_scope == "new_vs_new" and pair.severity == "exact"
    ]
    near_pairs = [
        pair
        for pair in pairs
        if pair.comparison_scope == "new_vs_new" and pair.severity == "near_duplicate"
    ]
    brownfield_pairs = [pair for pair in pairs if pair.comparison_scope == "new_vs_existing"]

    assert len(exact_pairs) == 1
    assert {exact_pairs[0].left_chunk.id, exact_pairs[0].right_chunk.id} == {"a", "b"}
    assert any({"a", "c"} == {pair.left_chunk.id, pair.right_chunk.id} for pair in near_pairs)
    assert any(pair.left_chunk.id == "a" and pair.right_chunk.id == "e" for pair in brownfield_pairs)


def test_duplication_pass_clusters_exact_duplicates_and_keeps_other_findings() -> None:
    """Duplication findings should cluster exact matches and preserve other review cases."""
    a = _chunk("a", "Exact text one", "one.txt", [1.0, 0.0])
    b = _chunk("b", "Exact text one copy", "two.txt", [1.0, 0.0])
    c = _chunk("c", "Exact text one copy two", "three.txt", [1.0, 0.0])
    d = _chunk("d", "Near duplicate text", "four.txt", [0.9, 0.435889894])
    e = _chunk("e", "Already indexed text", "indexed.txt", [1.0, 0.0])

    pairs = [
        SimilarityPair(a, b, 0.999, "new_vs_new", "exact"),
        SimilarityPair(b, c, 0.998, "new_vs_new", "exact"),
        SimilarityPair(a, d, 0.9, "new_vs_new", "near_duplicate"),
        SimilarityPair(a, e, 0.99, "new_vs_existing", "exact"),
    ]

    findings = DuplicationDetectionPass().build_findings(pairs)

    exact_findings = [finding for finding in findings if finding.finding_type == "exact"]
    near_findings = [finding for finding in findings if finding.finding_type == "near_duplicate"]
    brownfield_findings = [finding for finding in findings if finding.finding_type == "already_in_index"]

    assert len(exact_findings) == 1
    assert {chunk.chunk_id for chunk in exact_findings[0].chunks_involved} == {"a", "b", "c"}
    assert exact_findings[0].similarity_score == 0.998

    assert len(near_findings) == 1
    assert [chunk.chunk_id for chunk in near_findings[0].chunks_involved] == ["a", "d"]

    assert len(brownfield_findings) == 1
    assert [chunk.source_scope for chunk in brownfield_findings[0].chunks_involved] == ["new", "existing"]


def _chunk(
    chunk_id: str,
    text: str,
    filename: str,
    embedding: list[float],
) -> Chunk:
    """Build a chunk for similarity and duplication tests."""
    return Chunk(
        id=chunk_id,
        text=text,
        parent_document_id=f"doc-{chunk_id}",
        parent_document_name=filename,
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=len(text)),
        metadata={},
        embedding=embedding,
    )
