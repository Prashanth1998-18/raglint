from __future__ import annotations

import pytest

from app.models.document import Chunk, ChunkPosition
from app.services.similarity import SimilarityPair
from benchmarks.scripts.run_audits import _finding_counts, _flatten_findings, _staleness_subcategory
from benchmarks.scripts.validate_findings import _prepare_validation_findings
from benchmarks.utils.raglint_analysis import _cross_document_pairs
from benchmarks.utils.runtime import _version_key, ensure_min_package_version


def test_version_key_parses_numeric_prefix_only() -> None:
    assert _version_key("4.4.2") == (4, 4, 2)
    assert _version_key("4.4.2.dev0") == (4, 4, 2)
    assert _version_key("3.6.0") < _version_key("4.4.0")


def test_ensure_min_package_version_rejects_old_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks.utils.runtime.version", lambda package: "3.6.0")

    with pytest.raises(RuntimeError, match="too old"):
        ensure_min_package_version(
            "datasets",
            "4.4.0",
            install_command="pip install -r benchmarks/requirements.txt",
        )


def test_cross_document_pairs_drop_same_document_matches() -> None:
    left = _chunk("left", "doc-1", "a.txt")
    right_same = _chunk("right-same", "doc-1", "a.txt")
    right_other = _chunk("right-other", "doc-2", "b.txt")

    filtered = _cross_document_pairs(
        [
            SimilarityPair(left, right_same, 0.99, "new_vs_new", "exact"),
            SimilarityPair(left, right_other, 0.91, "new_vs_new", "near_duplicate"),
        ]
    )

    assert len(filtered) == 1
    assert filtered[0].right_chunk.id == "right-other"


def test_staleness_subcategory_separates_missing_signals_from_true_staleness() -> None:
    freshness_unknown = {
        "staleness_score": 0.5,
        "signals": {
            "metadata_reference_date": None,
            "content_reference_date": None,
            "detected_dates": [],
            "temporal_language": [],
            "version_references": [],
        },
    }
    signaled_stale = {
        "staleness_score": 0.8,
        "signals": {
            "metadata_reference_date": None,
            "content_reference_date": "2020-01-01T00:00:00+00:00",
            "detected_dates": ["As of January 2020"],
            "temporal_language": [],
            "version_references": [],
        },
    }

    assert _staleness_subcategory(freshness_unknown) == "freshness_unknown"
    assert _staleness_subcategory(signaled_stale) == "high"


def test_rot_outdated_is_excluded_when_same_document_is_already_stale() -> None:
    report_payload = {
        "duplication_findings": [],
        "contradiction_findings": [],
        "metadata_findings": [],
        "staleness_findings": [
            {
                "id": "stale-1",
                "staleness_score": 0.9,
                "signals": {
                    "metadata_reference_date": None,
                    "content_reference_date": "2020-01-01T00:00:00+00:00",
                    "detected_dates": ["January 2020"],
                    "temporal_language": [],
                    "version_references": [],
                },
                "chunk": {
                    "parent_document_id": "doc-1",
                    "source_filename": "doc-1.txt",
                    "text": "As of January 2020, the guidance remains in effect.",
                },
            }
        ],
        "rot_findings": [
            {
                "id": "rot-1",
                "document_id": "doc-1",
                "document_filename": "doc-1.txt",
                "classifications": ["outdated"],
            },
            {
                "id": "rot-2",
                "document_id": "doc-2",
                "document_filename": "doc-2.txt",
                "classifications": ["trivial"],
            },
        ],
    }

    counts = _finding_counts(report_payload)
    flattened = _flatten_findings("demo", report_payload)

    assert counts["stale_high"] == 1
    assert counts["rot_outdated"] == 0
    assert counts["rot_trivial"] == 1
    assert [finding["category"] for finding in flattened] == ["staleness", "rot"]
    assert [finding["subcategory"] for finding in flattened] == ["high", "trivial"]


def test_prepare_validation_findings_keeps_all_distinct_findings_but_drops_rot_outdated_overlap() -> None:
    findings = [
        {
            "dataset_name": "demo",
            "category": "staleness",
            "subcategory": "medium",
            "finding_id": "stale-low",
            "payload": {
                "staleness_score": 0.6,
                "chunk": {
                    "parent_document_id": "doc-1",
                    "source_filename": "doc-1.txt",
                },
            },
        },
        {
            "dataset_name": "demo",
            "category": "staleness",
            "subcategory": "high",
            "finding_id": "stale-high",
            "payload": {
                "staleness_score": 0.95,
                "chunk": {
                    "parent_document_id": "doc-1",
                    "source_filename": "doc-1.txt",
                },
            },
        },
        {
            "dataset_name": "demo",
            "category": "rot",
            "subcategory": "outdated",
            "finding_id": "rot-outdated-overlap",
            "payload": {
                "document_id": "doc-1",
                "document_filename": "doc-1.txt",
                "classifications": ["outdated"],
            },
        },
        {
            "dataset_name": "demo",
            "category": "rot",
            "subcategory": "trivial",
            "finding_id": "rot-trivial",
            "payload": {
                "document_id": "doc-2",
                "document_filename": "doc-2.txt",
                "classifications": ["trivial"],
            },
        },
    ]

    prepared = _prepare_validation_findings(findings)

    assert len(prepared) == 3
    assert {finding["finding_id"] for finding in prepared} == {"stale-low", "stale-high", "rot-trivial"}


def _chunk(chunk_id: str, document_id: str, filename: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"chunk {chunk_id}",
        parent_document_id=document_id,
        parent_document_name=filename,
        position=ChunkPosition(chunk_index=0, start_char=0, end_char=10),
        metadata={},
        embedding=[1.0, 0.0],
    )
