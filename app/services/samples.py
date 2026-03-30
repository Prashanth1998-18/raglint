"""Helpers for loading built-in sample corpora and cached sample reports."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from app.models.document import Chunk, Document
from app.services.chunker import RecursiveCharacterChunker
from app.services.metadata import build_document_metadata


SampleCorpusId = Literal["hr-policies", "tech-docs", "product-kb"]

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


@dataclass(frozen=True, slots=True)
class SampleCorpusDefinition:
    """Static metadata for a built-in sample corpus."""

    sample_id: SampleCorpusId
    title: str
    description: str
    document_count: int
    dataset_filename: str
    report_filename: str
    icon_key: str

    @property
    def dataset_path(self) -> Path:
        """Return the absolute path to the corpus JSON file."""
        return SAMPLES_DIR / self.dataset_filename

    @property
    def report_path(self) -> Path:
        """Return the absolute path to the cached report JSON file."""
        return SAMPLES_DIR / self.report_filename

    @property
    def start_url(self) -> str:
        """Return the route used to start the sample report experience."""
        return f"/report/samples/{self.sample_id}/start"


@dataclass(frozen=True, slots=True)
class SampleCorpusData:
    """A parsed sample corpus ready to use in analysis or tests."""

    definition: SampleCorpusDefinition
    documents: list[Document]
    chunks: list[Chunk]


SAMPLE_CORPORA: tuple[SampleCorpusDefinition, ...] = (
    SampleCorpusDefinition(
        sample_id="hr-policies",
        title="HR Policies",
        description="9 documents with contradictory policies, outdated content, and duplicate guides.",
        document_count=9,
        dataset_filename="hr-policies.json",
        report_filename="hr-policies-report.json",
        icon_key="policies",
    ),
    SampleCorpusDefinition(
        sample_id="tech-docs",
        title="Technical Docs",
        description="11 documents with version mismatches, duplicate guides, and stale deployment instructions.",
        document_count=11,
        dataset_filename="tech-docs.json",
        report_filename="tech-docs-report.json",
        icon_key="technical",
    ),
    SampleCorpusDefinition(
        sample_id="product-kb",
        title="Product Knowledge Base",
        description="11 documents with conflicting pricing, duplicate onboarding guides, and outdated features.",
        document_count=11,
        dataset_filename="product-kb.json",
        report_filename="product-kb-report.json",
        icon_key="product",
    ),
)

SAMPLE_CORPORA_BY_ID: dict[SampleCorpusId, SampleCorpusDefinition] = {
    sample.sample_id: sample for sample in SAMPLE_CORPORA
}


def list_sample_corpora() -> tuple[SampleCorpusDefinition, ...]:
    """Return the built-in sample corpus definitions in display order."""
    return SAMPLE_CORPORA


def get_sample_definition(sample_id: str) -> SampleCorpusDefinition:
    """Return a sample definition or raise a descriptive error for invalid ids."""
    try:
        return SAMPLE_CORPORA_BY_ID[sample_id]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unknown sample corpus: {sample_id}") from exc


def load_sample_corpus(
    sample_id: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> SampleCorpusData:
    """Load a sample corpus and convert it into normalized documents and chunks."""
    definition = get_sample_definition(sample_id)
    entries = _load_sample_entries(definition.dataset_path)
    documents = [_sample_document(definition, entry) for entry in entries]
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [chunk for document in documents for chunk in chunker.chunk_document(document)]
    return SampleCorpusData(definition=definition, documents=documents, chunks=chunks)


def build_sample_uploads(sample_id: str) -> list[UploadFile]:
    """Build in-memory upload objects so sample corpora can reuse the upload parser."""
    definition = get_sample_definition(sample_id)
    uploads: list[UploadFile] = []
    for entry in _load_sample_entries(definition.dataset_path):
        payload = entry["text"].encode("utf-8")
        uploads.append(
            UploadFile(
                filename=entry["filename"],
                file=io.BytesIO(payload),
            )
        )
    return uploads


def load_sample_report(sample_id: str) -> dict[str, object] | None:
    """Load the cached sample report payload when it exists on disk."""
    definition = get_sample_definition(sample_id)
    if not definition.report_path.exists():
        return None
    payload = json.loads(definition.report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sample report {definition.report_path.name} must contain a JSON object.")
    return payload


def clear_sample_caches() -> None:
    """Clear memoized sample file reads. Useful for tests and precompute runs."""
    _load_sample_entries.cache_clear()


@lru_cache(maxsize=None)
def _load_sample_entries(dataset_path: Path) -> tuple[dict[str, str], ...]:
    """Read and validate a sample corpus JSON file."""
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Sample corpus {dataset_path.name} must contain a JSON array.")

    entries: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Sample corpus {dataset_path.name} entry {index} must be an object.")
        filename = item.get("filename")
        title = item.get("title")
        text = item.get("text")
        if not all(isinstance(value, str) and value.strip() for value in (filename, title, text)):
            raise ValueError(
                f"Sample corpus {dataset_path.name} entry {index} must include non-empty filename, title, and text fields."
            )
        entries.append(
            {
                "filename": filename.strip(),
                "title": title.strip(),
                "text": text.strip(),
            }
        )
    return tuple(entries)


def _sample_document(definition: SampleCorpusDefinition, entry: dict[str, str]) -> Document:
    """Convert a validated sample entry into the shared document model."""
    text = entry["text"]
    return Document(
        text=text,
        metadata=build_document_metadata(
            filename=entry["filename"],
            file_size=len(text.encode("utf-8")),
            embedded={
                "title": entry["title"],
                "sample_corpus": definition.title,
            },
        ),
        source_path=f"sample://{definition.sample_id}/{entry['filename']}",
    )
