"""Tests for Phase 3 staleness scoring and metadata audit logic."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models.document import Chunk, ChunkPosition, Document
from app.services.metadata import build_document_metadata
from app.services.passes.metadata import MetadataAuditPass
from app.services.passes.staleness import StalenessScoringPass
from app.services.similarity import SimilarityPair


def test_staleness_pass_extracts_supported_date_patterns() -> None:
    """Regex extraction should find explicit dates, temporal phrases, and versions."""
    pass_under_test = StalenessScoringPass(
        current_datetime=datetime(2026, 3, 15, tzinfo=UTC),
    )

    signals = pass_under_test.extract_content_signals(
        (
            "Updated in Q3 2023, as of January 2024. "
            "Historical reference FY2022 and rollout date 2021-03-15. "
            "Version 3.1 is currently active at this time."
        )
    )

    assert "Q3 2023" in signals["detected_dates"]
    assert "as of January 2024" in signals["detected_dates"]
    assert "FY2022" in signals["detected_dates"]
    assert "2021-03-15" in signals["detected_dates"]
    assert "currently" in signals["temporal_language"]
    assert "at this time" in signals["temporal_language"]
    assert "version 3.1" in signals["version_references"]


def test_staleness_pass_scores_chunks_and_flags_supersession() -> None:
    """Staleness scoring should weight metadata and content signals and detect newer replacements."""
    current_datetime = datetime(2026, 3, 15, tzinfo=UTC)
    pass_under_test = StalenessScoringPass(current_datetime=current_datetime)

    document = Document(
        id="doc-new",
        text="As of January 2026, the policy is current.",
        metadata=build_document_metadata(
            filename="policy-2026.md",
            file_size=120,
            modified_at=datetime(2026, 1, 20, tzinfo=UTC),
        ),
        source_path="policy-2026.md",
    )
    new_chunk = Chunk(
        id="chunk-new",
        text="As of January 2026, the policy is current.",
        parent_document_id="doc-new",
        parent_document_name="policy-2026.md",
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=43),
        metadata={},
    )
    stale_chunk = Chunk(
        id="chunk-existing",
        text="As of January 2023, the previous policy applies.",
        parent_document_id="existing-doc",
        parent_document_name="policy-2023.md",
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=48),
        metadata={"modified_at": "2023-01-10T00:00:00+00:00"},
    )
    unknown_chunk = Chunk(
        id="chunk-unknown",
        text="This note has no explicit dates at all.",
        parent_document_id="unknown-doc",
        parent_document_name="notes.txt",
        position=ChunkPosition(chunk_index=1, start_char=44, end_char=80),
        metadata={},
    )

    findings, supersession_findings = pass_under_test.build_findings(
        [new_chunk, unknown_chunk],
        [document],
        existing_chunks=[stale_chunk],
        similarity_pairs=[
            SimilarityPair(new_chunk, stale_chunk, 0.78, "new_vs_existing", "near_duplicate"),
        ],
    )

    assert len(findings) == 1
    unknown_finding = next(finding for finding in findings if finding.chunk.chunk_id == "chunk-unknown")

    assert unknown_finding.staleness_score == 0.5

    assert len(supersession_findings) == 1
    assert supersession_findings[0].finding_type == "potentially_superseded"
    assert supersession_findings[0].chunk.source_filename == "policy-2023.md"
    assert supersession_findings[0].related_chunk is not None
    assert supersession_findings[0].related_chunk.source_filename == "policy-2026.md"


def test_staleness_pass_raises_scores_for_old_content_even_with_fresh_file_metadata() -> None:
    """Older text references should remain stale even when the uploaded file metadata is fresh."""
    current_datetime = datetime(2026, 3, 15, tzinfo=UTC)
    pass_under_test = StalenessScoringPass(current_datetime=current_datetime)

    security_document = Document(
        id="doc-security",
        text=(
            "Last updated: March 15, 2021. "
            "As of Q1 2021, Jane Doe served as Chief Security Officer."
        ),
        metadata=build_document_metadata(
            filename="security-policy-2021.md",
            file_size=180,
            modified_at=datetime(2026, 2, 20, tzinfo=UTC),
        ),
        source_path="security-policy-2021.md",
    )
    meeting_document = Document(
        id="doc-meeting",
        text="Meeting notes from October 2023 cover migration owners and action items.",
        metadata=build_document_metadata(
            filename="meeting-notes-oct.md",
            file_size=120,
            modified_at=datetime(2026, 2, 18, tzinfo=UTC),
        ),
        source_path="meeting-notes-oct.md",
    )

    security_chunk = Chunk(
        id="chunk-security",
        text=security_document.text,
        parent_document_id="doc-security",
        parent_document_name="security-policy-2021.md",
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=len(security_document.text)),
        metadata={},
    )
    meeting_chunk = Chunk(
        id="chunk-meeting",
        text=meeting_document.text,
        parent_document_id="doc-meeting",
        parent_document_name="meeting-notes-oct.md",
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=len(meeting_document.text)),
        metadata={},
    )

    findings, _ = pass_under_test.build_findings(
        [security_chunk, meeting_chunk],
        [security_document, meeting_document],
    )

    by_chunk = {finding.chunk.chunk_id: finding for finding in findings}

    assert by_chunk["chunk-security"].staleness_score >= 0.8
    assert by_chunk["chunk-security"].recommendation == "Review or remove before indexing."
    assert by_chunk["chunk-meeting"].staleness_score >= 0.6
    assert by_chunk["chunk-meeting"].recommendation == "May be outdated, review recommended."


def test_metadata_audit_scores_completeness_and_summary() -> None:
    """Metadata audit should calculate per-document completeness and corpus summary fields."""
    sparse_document = Document(
        id="doc-sparse",
        text="Sparse metadata document.",
        metadata=build_document_metadata(
            filename="document1.pdf",
            file_size=64,
            modified_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        source_path="document1.pdf",
    )
    rich_document = Document(
        id="doc-rich",
        text="Rich metadata document.",
        metadata=build_document_metadata(
            filename="finance-policy.md",
            file_size=128,
            created_at=datetime(2025, 5, 1, tzinfo=UTC),
            modified_at=datetime(2026, 1, 5, tzinfo=UTC),
            embedded={"author": "Ops Team", "category": "Policy", "version": "3.1", "title": "Finance Policy"},
            frontmatter={"owner": "Ops Team", "type": "Policy"},
        ),
        source_path="finance-policy.md",
    )

    findings, summary = MetadataAuditPass().audit_documents([sparse_document, rich_document])

    sparse_finding = next(finding for finding in findings if finding.document_id == "doc-sparse")
    rich_finding = next(finding for finding in findings if finding.document_id == "doc-rich")

    assert sparse_finding.completeness_score == 0.167
    assert "title or meaningful filename" in sparse_finding.missing_fields
    assert "author or owner" in sparse_finding.missing_fields
    assert rich_finding.completeness_score == 1.0

    assert summary.average_completeness == 0.584
    assert summary.most_common_missing_fields["title or meaningful filename"] == 1


def test_metadata_audit_uses_configured_fields_and_custom_keys() -> None:
    """Metadata audit should respect user-configured field names instead of only the defaults."""
    document = Document(
        id="doc-legal",
        text="Legal memo.",
        metadata=build_document_metadata(
            filename="legal-memo.md",
            file_size=96,
            modified_at=datetime(2026, 1, 8, tzinfo=UTC),
            embedded={"case_number": "2026-17"},
            frontmatter={"owner": "Legal Ops"},
        ),
        source_path="legal-memo.md",
    )

    findings, summary = MetadataAuditPass(
        expected_fields=["title", "owner", "case number", "jurisdiction"],
    ).audit_documents([document])

    assert summary.expected_fields == ["title", "owner", "case number", "jurisdiction"]
    assert findings[0].completeness_score == 0.75
    assert findings[0].missing_fields == ["jurisdiction"]
