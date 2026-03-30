from __future__ import annotations

from benchmarks.scripts.download_datasets import (
    _extract_hotpotqa_documents,
    _extract_ms_marco_documents,
    _extract_squad_documents,
    _finalize_dataset,
)
from benchmarks.utils.config import DatasetAuditConfig


def test_extract_squad_documents_collects_context_rows() -> None:
    rows = [
        {"id": "1", "title": "Oxygen", "context": "Oxygen is a chemical element."},
        {"id": "2", "title": "Hydrogen", "context": "Hydrogen is the lightest element."},
    ]

    documents = _extract_squad_documents(rows)

    assert len(documents) == 2
    assert documents[0]["title"] == "Oxygen"
    assert documents[0]["metadata"]["original_id"] == "1"
    assert documents[1]["text"] == "Hydrogen is the lightest element."


def test_extract_hotpotqa_documents_uses_supporting_context_only() -> None:
    rows = [
        {
            "id": "hp-1",
            "supporting_facts": [["Mars", 0], ["Phobos", 1]],
            "context": [
                ["Mars", ["Mars is the fourth planet from the Sun.", "It has two moons."]],
                ["Phobos", ["Phobos orbits Mars."]],
                ["Venus", ["Venus is the second planet from the Sun."]],
            ],
        }
    ]

    documents = _extract_hotpotqa_documents(rows)

    assert len(documents) == 2
    assert {document["title"] for document in documents} == {"Mars", "Phobos"}
    assert all(document["metadata"]["original_id"] == "hp-1" for document in documents)


def test_extract_ms_marco_documents_reads_passage_text_and_url() -> None:
    rows = [
        {
            "query_id": "mm-1",
            "passages": {
                "passage_text": ["First passage.", "Second passage."],
                "url": ["https://example.com/1", "https://example.com/2"],
            },
        }
    ]

    documents = _extract_ms_marco_documents(rows)

    assert len(documents) == 2
    assert documents[0]["metadata"]["original_id"] == "mm-1"
    assert documents[1]["metadata"]["url"] == "https://example.com/2"


def test_finalize_dataset_deduplicates_and_standardizes_documents() -> None:
    dataset_config = DatasetAuditConfig(
        name="squad_v2",
        source="rajpurkar/squad_v2",
        subset_size=None,
        split="validation",
    )
    raw_documents = [
        {"title": "Oxygen", "text": "Oxygen is a chemical element.", "metadata": {"original_id": "1"}},
        {"title": "Oxygen", "text": "Oxygen is a chemical element.", "metadata": {"original_id": "2"}},
        {"title": "Hydrogen", "text": "Hydrogen is the lightest element.", "metadata": {"original_id": "3"}},
    ]

    prepared = _finalize_dataset(
        dataset_config=dataset_config,
        raw_documents=raw_documents,
        seed=42,
        loaded_source="rajpurkar/squad_v2",
    )

    assert prepared.dataset_name == "squad_v2"
    assert prepared.document_count == 2
    assert prepared.metadata["duplicates_removed"] == 1
    assert prepared.documents[0].filename == "squad_v2-0001.txt"
    assert prepared.documents[0].metadata["source_dataset"] == "squad_v2"
