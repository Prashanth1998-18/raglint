from __future__ import annotations

import json
from pathlib import Path
import shutil

from benchmarks.scripts.generate_report import generate_benchmark_report


def test_generate_benchmark_report_aggregates_audits_and_validation() -> None:
    output_dir = _make_results_dir("aggregate")
    try:
        _write_audit(
            output_dir / "squad_v2_audit.json",
            dataset_name="squad_v2",
            source="rajpurkar/squad_v2",
            document_count=5,
            chunk_count=8,
            health_score=72.5,
            duplicates=2,
            contradictions=1,
            stale=1,
            metadata=3,
            rot=1,
            documents_with_issues=3,
        )
        _write_audit(
            output_dir / "hotpotqa_audit.json",
            dataset_name="hotpotqa",
            source="hotpot_qa",
            document_count=7,
            chunk_count=12,
            health_score=68.0,
            duplicates=1,
            contradictions=2,
            stale=2,
            metadata=1,
            rot=2,
            documents_with_issues=4,
        )
        (output_dir / "validation_summary.json").write_text(
            json.dumps(
                {
                    "total_sampled": 10,
                    "total_valid": 8,
                    "overall_precision": 80.0,
                    "precision_by_category": {
                        "contradiction": {"sampled": 4, "valid": 4, "precision": 100.0},
                        "duplication": {"sampled": 3, "valid": 2, "precision": 66.67},
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        payload = generate_benchmark_report(output_dir=output_dir)

        assert payload["headline_numbers"]["dataset_count"] == 2
        assert payload["headline_numbers"]["total_documents"] == 12
        assert payload["headline_numbers"]["manual_validation_precision"] == 80.0
        assert payload["headline_numbers"]["total_issue_instances"] == 2
        assert payload["datasets"][0]["display_name"] == "HotpotQA"
        assert (output_dir / "benchmark_report.json").exists()
        markdown = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        assert "Manual validation reviewed 10 findings." in markdown
        assert "| SQuAD 2.0 | 5 | 72.5 |" in markdown
        assert "Metadata Gaps" not in markdown
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_generate_benchmark_report_mentions_pending_validation_when_missing() -> None:
    output_dir = _make_results_dir("pending-validation")
    try:
        _write_audit(
            output_dir / "techqa_audit.json",
            dataset_name="techqa",
            source="ibm/techqa",
            document_count=3,
            chunk_count=4,
            health_score=91.0,
            duplicates=0,
            contradictions=0,
            stale=0,
            metadata=1,
            rot=0,
            documents_with_issues=1,
        )

        payload = generate_benchmark_report(output_dir=output_dir)

        assert payload["validation"] is None
        assert payload["headline_numbers"]["manual_validation_precision"] is None
        markdown = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        assert "Manual validation has not been summarized yet." in markdown
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_generate_benchmark_report_treats_zero_sample_validation_as_pending() -> None:
    output_dir = _make_results_dir("zero-sample-validation")
    try:
        _write_audit(
            output_dir / "techqa_audit.json",
            dataset_name="techqa",
            source="ibm/techqa",
            document_count=3,
            chunk_count=4,
            health_score=91.0,
            duplicates=0,
            contradictions=0,
            stale=0,
            metadata=1,
            rot=0,
            documents_with_issues=1,
        )
        (output_dir / "validation_summary.json").write_text(
            json.dumps(
                {
                    "total_sampled": 0,
                    "total_valid": 0,
                    "overall_precision": 0.0,
                    "precision_by_category": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        payload = generate_benchmark_report(output_dir=output_dir)

        assert payload["validation"] is None
        assert payload["headline_numbers"]["manual_validation_precision"] is None
        markdown = (output_dir / "benchmark_report.md").read_text(encoding="utf-8")
        assert "Manual validation has not been summarized yet." in markdown
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _make_results_dir(name: str) -> Path:
    output_dir = Path("benchmarks") / "results" / "test-artifacts" / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_audit(
    path: Path,
    *,
    dataset_name: str,
    source: str,
    document_count: int,
    chunk_count: int,
    health_score: float,
    duplicates: int,
    contradictions: int,
    stale: int,
    metadata: int,
    rot: int,
    documents_with_issues: int,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "source": source,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "health_score": health_score,
        "dimension_scores": {
            "duplication": 80.0,
            "staleness": 70.0,
            "contradictions": 60.0,
            "metadata": None,
            "rot": 90.0,
        },
        "raw_dimension_scores": {
            "duplication": 80.0,
            "staleness": 70.0,
            "contradictions": 60.0,
            "metadata": 50.0,
            "rot": 90.0,
        },
        "finding_counts": {
            "duplicates_exact": duplicates,
            "duplicates_near": 0,
            "contradictions": contradictions,
            "stale_high": stale,
            "stale_medium": 0,
            "freshness_unknown": 4,
            "metadata_poor": metadata,
            "rot_trivial": rot,
            "rot_redundant": 0,
            "rot_outdated": 0,
        },
        "findings": [
            {
                "dataset_name": dataset_name,
                "category": "contradiction",
                "subcategory": "high",
                "finding_id": f"{dataset_name}-contradiction-1",
                "payload": {
                    "severity": "high",
                    "similarity_score": 0.88,
                    "claim_a": "The API ships in April.",
                    "claim_b": "The API ships in June.",
                    "chunks_involved": [
                        {"source_filename": "left.txt", "text": "The API ships in April.", "parent_document_id": f"{dataset_name}-doc-1"},
                        {"source_filename": "right.txt", "text": "The API ships in June.", "parent_document_id": f"{dataset_name}-doc-2"},
                    ],
                },
            }
        ],
        "report_payload": {
            "documents": [
                {"id": f"{dataset_name}-doc-1", "text": "The API ships in April."},
            ],
            "health_score": {
                "summary": {
                    "unique_documents_with_issues": documents_with_issues,
                    "documents_recommended_for_removal": duplicates + rot,
                }
            },
        },
        "dataset_metadata": {"duplicates_removed": 2},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
