"""Tests for Phase 5 ROT classification, scoring, and exports."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models.document import (
    Chunk,
    ChunkPosition,
    ContradictionRunStats,
    Document,
    DuplicationFinding,
    FindingChunk,
    MetadataAuditSummary,
    MetadataFinding,
    ROTFinding,
    StalenessFinding,
)
from app.services.metadata import build_document_metadata
from app.services.passes.rot import ROTClassificationPass
from app.services.scoring import CorpusHealthScorer


def test_rot_pass_classifies_documents_from_duplication_staleness_and_triviality_signals() -> None:
    """ROT should classify redundant, outdated, trivial, and healthy documents correctly."""
    redundant_document = _document("doc-redundant", "redundant.md", "Alpha text.\n\nAlpha text copy.")
    outdated_document = _document("doc-outdated", "outdated.md", "As of January 2020, the policy is unchanged.")
    trivial_document = _document("doc-trivial", "trivial.md", "TODO placeholder draft agenda")
    healthy_document = _document(
        "doc-healthy",
        "healthy.md",
        "The deployment runbook explains rollback steps, ownership, and escalation paths for production incidents.",
    )

    redundant_chunk_a = _chunk("chunk-a", "Alpha text.", "doc-redundant", "redundant.md")
    redundant_chunk_b = _chunk("chunk-b", "Alpha text copy.", "doc-redundant", "redundant.md")
    outdated_chunk = _chunk("chunk-c", outdated_document.text, "doc-outdated", "outdated.md")
    trivial_chunk = _chunk("chunk-d", trivial_document.text, "doc-trivial", "trivial.md")
    healthy_chunk = _chunk("chunk-e", healthy_document.text, "doc-healthy", "healthy.md")

    duplication_findings = [
        DuplicationFinding(
            finding_type="exact",
            chunks_involved=[
                FindingChunk(
                    chunk_id="chunk-a",
                    text=redundant_chunk_a.text,
                    source_filename="redundant.md",
                    parent_document_id="doc-redundant",
                    source_scope="new",
                ),
                FindingChunk(
                    chunk_id="chunk-b",
                    text=redundant_chunk_b.text,
                    source_filename="redundant.md",
                    parent_document_id="doc-redundant",
                    source_scope="new",
                ),
            ],
            similarity_score=0.995,
            explanation="Duplicate cluster.",
            recommendation="Keep one.",
        )
    ]
    staleness_findings = [
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(
                chunk_id="chunk-c",
                text=outdated_chunk.text,
                source_filename="outdated.md",
                parent_document_id="doc-outdated",
                source_scope="new",
            ),
            staleness_score=0.91,
            signals={},
            explanation="Old date references.",
            recommendation="Update this document.",
        ),
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(
                chunk_id="chunk-e",
                text=healthy_chunk.text,
                source_filename="healthy.md",
                parent_document_id="doc-healthy",
                source_scope="new",
            ),
            staleness_score=0.15,
            signals={},
            explanation="Fresh enough.",
            recommendation="Keep.",
        ),
    ]

    findings = _run(
        ROTClassificationPass().run(
            [
                redundant_document,
                outdated_document,
                trivial_document,
                healthy_document,
            ],
            [
                redundant_chunk_a,
                redundant_chunk_b,
                outdated_chunk,
                trivial_chunk,
                healthy_chunk,
            ],
            duplication_findings=duplication_findings,
            staleness_findings=staleness_findings,
            supersession_findings=[],
            contradiction_findings=[],
            metadata_findings=[],
        )
    )

    by_document = {finding.document_id: finding for finding in findings}

    assert by_document["doc-redundant"].classifications == ["redundant"]
    assert by_document["doc-redundant"].signals["duplication"]["duplicate_ratio"] == 1.0

    assert by_document["doc-outdated"].classifications == ["outdated"]
    assert by_document["doc-outdated"].signals["staleness"]["average_staleness_score"] == 0.91

    assert by_document["doc-trivial"].classifications == ["trivial"]
    assert by_document["doc-trivial"].signals["triviality"]["average_chunk_word_count"] == 4.0

    assert by_document["doc-healthy"].classifications == ["healthy"]


def test_corpus_health_scorer_calculates_dimension_scores_and_handles_na_contradictions() -> None:
    """Phase 5 scoring should compute weighted health metrics and contradiction N/A cases."""
    documents = [
        _document("doc-one", "one.md", "Document one."),
        _document("doc-two", "two.md", "Document two."),
    ]
    chunks = [
        _chunk("a", "alpha", "doc-one", "one.md"),
        _chunk("b", "beta", "doc-one", "one.md"),
        _chunk("c", "gamma", "doc-two", "two.md"),
        _chunk("d", "delta", "doc-two", "two.md"),
    ]
    duplication_findings = [
        DuplicationFinding(
            finding_type="exact",
            chunks_involved=[
                FindingChunk(chunk_id="a", text="alpha", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
                FindingChunk(chunk_id="b", text="beta", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
            ],
            similarity_score=0.99,
            explanation="Duplicate cluster.",
            recommendation="Remove duplicates.",
        ),
        DuplicationFinding(
            finding_type="near_duplicate",
            chunks_involved=[
                FindingChunk(chunk_id="a", text="alpha", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
                FindingChunk(chunk_id="b", text="beta", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
            ],
            similarity_score=0.87,
            explanation="Near-duplicate pair.",
            recommendation="Review duplicates.",
        ),
    ]
    staleness_findings = [
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(chunk_id="a", text="alpha", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
            staleness_score=0.8,
            signals={},
            explanation="Old.",
            recommendation="Update or remove.",
        ),
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(chunk_id="b", text="beta", source_filename="one.md", parent_document_id="doc-one", source_scope="new"),
            staleness_score=0.2,
            signals={},
            explanation="Fine.",
            recommendation="Keep.",
        ),
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(chunk_id="c", text="gamma", source_filename="two.md", parent_document_id="doc-two", source_scope="new"),
            staleness_score=0.0,
            signals={},
            explanation="Fine.",
            recommendation="Keep.",
        ),
        StalenessFinding(
            finding_type="staleness",
            chunk=FindingChunk(chunk_id="d", text="delta", source_filename="two.md", parent_document_id="doc-two", source_scope="new"),
            staleness_score=0.4,
            signals={},
            explanation="Fine.",
            recommendation="Keep.",
        ),
    ]
    rot_findings = [
        ROTFinding(
            document_id="doc-one",
            document_filename="one.md",
            classifications=["redundant"],
            signals={"metadata": {"completeness_score": 0.4}},
            explanation="Duplicate-heavy.",
            impact_on_rag_quality="Crowds retrieval.",
            recommendation="Remove this document.",
        ),
        ROTFinding(
            document_id="doc-two",
            document_filename="two.md",
            classifications=["healthy"],
            signals={"metadata": {"completeness_score": 0.8}},
            explanation="Healthy.",
            impact_on_rag_quality="Low risk.",
            recommendation="Review complete. Keep this document in the corpus.",
        ),
    ]
    metadata_findings = [
        MetadataFinding(
            document_id="doc-one",
            document_filename="one.md",
            completeness_score=0.4,
            missing_fields=["author"],
            consistency_issues=[],
            explanation="Missing ownership metadata.",
            recommendation="Add author metadata.",
        ),
        MetadataFinding(
            document_id="doc-two",
            document_filename="two.md",
            completeness_score=0.8,
            missing_fields=[],
            consistency_issues=[],
            explanation="Metadata is mostly complete.",
            recommendation="Keep.",
        ),
    ]

    scorer = CorpusHealthScorer()
    score = scorer.calculate(
        documents=documents,
        chunks=chunks,
        duplication_findings=duplication_findings,
        staleness_findings=staleness_findings,
        contradiction_findings=[],
        contradiction_stats=ContradictionRunStats(candidate_pairs_considered=0),
        metadata_findings=metadata_findings,
        metadata_summary=MetadataAuditSummary(average_completeness=0.6),
        rot_findings=rot_findings,
        contradiction_score_available=False,
    )

    assert score.duplication_score == 50.0
    assert score.staleness_score == 65.0
    assert score.contradiction_score is None
    assert score.metadata_score == 60.0
    assert score.rot_score == 50.0
    assert score.overall_score == 55.3
    assert score.summary.total_chunks_analyzed == 4
    assert score.summary.unique_documents_with_issues == 2
    assert score.summary.duplicate_clusters == 2
    assert score.summary.documents_recommended_for_removal == 1
    assert score.projected_overall_score >= score.overall_score


def test_report_exports_reuse_cached_session_results() -> None:
    """The export endpoints should return the last preview report without rerunning analysis."""
    client = TestClient(app)

    preview_response = client.post(
        "/report/preview",
        data={"chunk_size": "80", "chunk_overlap": "10"},
        files=[
            ("documents", ("rot-note.txt", b"TODO placeholder draft agenda", "text/plain")),
        ],
    )

    assert preview_response.status_code == 200

    json_response = client.post("/report/export/json")
    markdown_response = client.post("/report/export/markdown")

    assert json_response.status_code == 200
    assert "attachment; filename=\"raglint-report.json\"" == json_response.headers["content-disposition"]
    json_payload = json_response.json()
    assert "health_score" in json_payload
    assert "rot_findings" in json_payload

    assert markdown_response.status_code == 200
    assert "attachment; filename=\"raglint-report.md\"" == markdown_response.headers["content-disposition"]
    assert "# RAGLint Report" in markdown_response.text
    assert "## Health Score" in markdown_response.text


def _document(document_id: str, filename: str, text: str) -> Document:
    """Build a document for Phase 5 tests."""
    return Document(
        id=document_id,
        text=text,
        metadata=build_document_metadata(filename=filename, file_size=len(text.encode("utf-8"))),
        source_path=filename,
    )


def _chunk(chunk_id: str, text: str, document_id: str, filename: str) -> Chunk:
    """Build a chunk for Phase 5 tests."""
    return Chunk(
        id=chunk_id,
        text=text,
        parent_document_id=document_id,
        parent_document_name=filename,
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=len(text)),
        metadata={},
    )


def _run(awaitable):
    """Run a single coroutine without introducing a pytest async dependency."""
    import asyncio

    return asyncio.run(awaitable)
