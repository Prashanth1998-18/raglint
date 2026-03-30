"""Application models for parsed documents and normalized chunks."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _generate_id() -> str:
    """Create a stable unique identifier for in-memory objects."""
    return uuid4().hex


class DocumentMetadata(BaseModel):
    """Metadata extracted from an uploaded document."""

    filename: str
    extension: str
    file_size: int
    created_at: datetime | None = None
    modified_at: datetime | None = None
    embedded: dict[str, object] = Field(default_factory=dict)
    frontmatter: dict[str, object] = Field(default_factory=dict)


class Document(BaseModel):
    """A parsed document ready for chunking and later analysis."""

    id: str = Field(default_factory=_generate_id)
    text: str
    metadata: DocumentMetadata
    source_path: str


class ChunkPosition(BaseModel):
    """Character offsets for a chunk within its parent text."""

    chunk_index: int
    start_char: int
    end_char: int


class Chunk(BaseModel):
    """A normalized chunk from either a parsed document or an import export."""

    id: str = Field(default_factory=_generate_id)
    text: str
    parent_document_id: str | None = None
    parent_document_name: str | None = None
    position: ChunkPosition
    metadata: dict[str, object] = Field(default_factory=dict)
    embedding: list[float] | None = None


class FindingChunk(BaseModel):
    """Chunk details included in report findings."""

    chunk_id: str
    text: str
    source_filename: str
    parent_document_id: str | None = None
    source_scope: str


class DuplicationFinding(BaseModel):
    """Human-readable duplication finding for the report view."""

    id: str = Field(default_factory=_generate_id)
    finding_type: Literal["exact", "near_duplicate", "already_in_index"]
    chunks_involved: list[FindingChunk]
    similarity_score: float
    explanation: str
    recommendation: str


class StalenessFinding(BaseModel):
    """A staleness finding for a chunk or an existing chunk supersession case."""

    id: str = Field(default_factory=_generate_id)
    finding_type: Literal["staleness", "potentially_superseded"]
    chunk: FindingChunk
    staleness_score: float
    signals: dict[str, object] = Field(default_factory=dict)
    explanation: str
    recommendation: str
    related_chunk: FindingChunk | None = None


class MetadataFinding(BaseModel):
    """A per-document metadata completeness and consistency finding."""

    id: str = Field(default_factory=_generate_id)
    document_id: str
    document_filename: str
    completeness_score: float
    missing_fields: list[str] = Field(default_factory=list)
    consistency_issues: list[str] = Field(default_factory=list)
    explanation: str
    recommendation: str


class MetadataAuditSummary(BaseModel):
    """Corpus-level metadata audit summary data."""

    average_completeness: float
    most_common_missing_fields: dict[str, int] = Field(default_factory=dict)
    consistency_issues: list[str] = Field(default_factory=list)
    expected_fields: list[str] = Field(default_factory=list)


class ContradictionFinding(BaseModel):
    """A contradiction finding produced from two similar chunks."""

    id: str = Field(default_factory=_generate_id)
    chunks_involved: list[FindingChunk]
    similarity_score: float
    severity: Literal["high", "medium"]
    claim_a: str
    claim_b: str
    explanation: str
    why_it_matters: str
    recommendation: str


class ContradictionRunStats(BaseModel):
    """Usage and cost summary for contradiction detection."""

    llm_calls_made: int = 0
    failed_calls: int = 0
    candidate_pairs_considered: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ROTFinding(BaseModel):
    """A document-level ROT classification built from prior pass outputs."""

    id: str = Field(default_factory=_generate_id)
    document_id: str
    document_filename: str
    classifications: list[Literal["healthy", "redundant", "outdated", "trivial"]] = Field(default_factory=list)
    signals: dict[str, object] = Field(default_factory=dict)
    explanation: str
    impact_on_rag_quality: str
    recommendation: str


class HealthDimensionScore(BaseModel):
    """A named score for one dimension of corpus health."""

    key: Literal["duplication", "staleness", "contradictions", "metadata", "rot"]
    label: str
    score: float | None = None
    weight: float


class CorpusHealthSummary(BaseModel):
    """Summary statistics used by the final report dashboard."""

    total_chunks_analyzed: int
    total_documents: int
    unique_documents_with_issues: int
    duplicate_clusters: int
    stale_chunks: int
    contradictions_found: int
    average_metadata_completeness: float
    documents_recommended_for_removal: int


class CorpusHealthScore(BaseModel):
    """Overall corpus health with per-dimension scores and summary counts."""

    overall_score: float
    projected_overall_score: float
    duplication_score: float | None = None
    staleness_score: float | None = None
    contradiction_score: float | None = None
    metadata_score: float | None = None
    rot_score: float | None = None
    dimension_scores: list[HealthDimensionScore] = Field(default_factory=list)
    skipped_dimensions: list[str] = Field(default_factory=list)
    summary: CorpusHealthSummary
