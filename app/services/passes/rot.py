"""ROT classification pass for Phase 5."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.models.document import (
    Chunk,
    ContradictionFinding,
    Document,
    DuplicationFinding,
    MetadataFinding,
    ROTFinding,
    StalenessFinding,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrivialityChunkAssessment:
    """Per-chunk information density signals for triviality checks."""

    chunk: Chunk
    word_count: int
    unique_ratio: float
    boilerplate_hits: list[str]
    boilerplate_ratio: float
    is_trivial: bool
    is_borderline: bool
    llm_verdict: str | None = None


@dataclass(slots=True)
class TrivialityDocumentAssessment:
    """Document-level triviality assessment built from chunk heuristics."""

    is_trivial: bool
    average_word_count: float
    average_unique_ratio: float
    average_boilerplate_ratio: float
    trivial_chunk_ratio: float
    boilerplate_patterns: list[str]
    llm_checks_used: int
    llm_supported_triviality: bool


class ROTClassificationPass:
    """Aggregate prior pass results into document-level ROT classifications."""

    WORD_PATTERN = re.compile(r"\b[a-z0-9][a-z0-9'\-]*\b", re.IGNORECASE)
    BOILERPLATE_PATTERNS = (
        "TBD",
        "TODO",
        "placeholder",
        "draft",
        "N/A",
        "to be determined",
        "agenda",
        "action items: none",
    )
    SUBSTANCE_SYSTEM_PROMPT = (
        "You are reviewing one document chunk for retrieval value in a RAG system. "
        "Decide whether the chunk contains substantive, retrievable information that could help answer user "
        "questions. Boilerplate, placeholders, agendas without decisions, and empty drafts are not substantive. "
        'Respond only as JSON with {"status":"SUBSTANTIVE"} or {"status":"TRIVIAL"}.'
    )

    def __init__(
        self,
        *,
        max_llm_checks: int = 3,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.max_llm_checks = max_llm_checks
        self.llm_model = llm_model

    async def run(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        *,
        duplication_findings: list[DuplicationFinding],
        staleness_findings: list[StalenessFinding],
        supersession_findings: list[StalenessFinding],
        contradiction_findings: list[ContradictionFinding],
        metadata_findings: list[MetadataFinding],
        api_key: str | None = None,
    ) -> list[ROTFinding]:
        """Classify each uploaded document as healthy, redundant, outdated, trivial, or a combination."""
        if not documents:
            return []

        document_lookup = {document.id: document for document in documents}
        chunks_by_document: dict[str, list[Chunk]] = {document.id: [] for document in documents}
        for chunk in chunks:
            if chunk.parent_document_id in chunks_by_document:
                chunks_by_document[chunk.parent_document_id].append(chunk)

        duplicated_chunk_ids, duplication_stats = self._duplication_signals(duplication_findings, document_lookup)
        staleness_by_document = self._staleness_signals(staleness_findings)
        supersession_by_document = self._supersession_signals(supersession_findings)
        contradiction_by_document = self._contradiction_signals(contradiction_findings)
        metadata_by_document = {finding.document_id: finding for finding in metadata_findings}

        llm_checks_remaining = self.max_llm_checks if api_key else 0
        findings: list[ROTFinding] = []

        for document in documents:
            document_chunks = chunks_by_document.get(document.id, [])
            document_chunk_ids = {chunk.id for chunk in document_chunks}
            duplication_signal = duplication_stats.get(
                document.id,
                {"exact": 0, "near_duplicate": 0, "already_in_index": 0},
            )
            duplicated_count = len(document_chunk_ids & duplicated_chunk_ids)
            total_chunks = max(len(document_chunks), 1)
            duplicate_ratio = duplicated_count / total_chunks

            document_staleness_findings = staleness_by_document.get(document.id, [])
            average_staleness = round(
                sum(finding.staleness_score for finding in document_staleness_findings) / len(document_staleness_findings),
                3,
            ) if document_staleness_findings else 0.0
            superseded_count = len(supersession_by_document.get(document.id, []))

            triviality = await self._assess_triviality(
                document_chunks,
                api_key=api_key,
                llm_checks_remaining=llm_checks_remaining,
            )
            llm_checks_remaining = max(llm_checks_remaining - triviality.llm_checks_used, 0)

            contradiction_count = len(contradiction_by_document.get(document.id, []))
            metadata_finding = metadata_by_document.get(document.id)

            classifications: list[str] = []
            if duplicate_ratio >= 0.5 or (duplicated_count > 0 and len(document_chunks) <= 1):
                classifications.append("redundant")
            if average_staleness > 0.7 or superseded_count > 0:
                classifications.append("outdated")
            if triviality.is_trivial:
                classifications.append("trivial")
            if not classifications:
                classifications.append("healthy")

            signals = {
                "duplication": {
                    "duplicated_chunks": duplicated_count,
                    "total_chunks": len(document_chunks),
                    "duplicate_ratio": round(duplicate_ratio, 3),
                    "exact_duplicates": duplication_signal["exact"],
                    "near_duplicates": duplication_signal["near_duplicate"],
                    "already_in_index": duplication_signal["already_in_index"],
                },
                "staleness": {
                    "average_staleness_score": average_staleness,
                    "stale_chunks_above_0_7": sum(
                        1
                        for finding in document_staleness_findings
                        if finding.staleness_score > 0.7
                    ),
                    "highest_chunk_staleness": round(
                        max((finding.staleness_score for finding in document_staleness_findings), default=0.0),
                        3,
                    ),
                    "superseded_matches": superseded_count,
                },
                "triviality": {
                    "average_chunk_word_count": round(triviality.average_word_count, 1),
                    "average_unique_term_ratio": round(triviality.average_unique_ratio, 3),
                    "average_boilerplate_ratio": round(triviality.average_boilerplate_ratio, 3),
                    "trivial_chunk_ratio": round(triviality.trivial_chunk_ratio, 3),
                    "boilerplate_patterns": triviality.boilerplate_patterns,
                    "llm_checks_used": triviality.llm_checks_used,
                    "llm_supported_triviality": triviality.llm_supported_triviality,
                },
                "contradictions": {
                    "count": contradiction_count,
                },
            }
            if metadata_finding is not None:
                signals["metadata"] = {
                    "completeness_score": metadata_finding.completeness_score,
                    "missing_fields": metadata_finding.missing_fields,
                }

            findings.append(
                ROTFinding(
                    document_id=document.id,
                    document_filename=document.metadata.filename,
                    classifications=classifications,
                    signals=signals,
                    explanation=self._build_explanation(
                        document.metadata.filename,
                        classifications,
                        duplicate_ratio=duplicate_ratio,
                        average_staleness=average_staleness,
                        triviality=triviality,
                    ),
                    impact_on_rag_quality=self._build_impact(classifications, contradiction_count),
                    recommendation=self._build_recommendation(classifications),
                )
            )

        findings.sort(
            key=lambda finding: (
                1 if "healthy" in finding.classifications else 0,
                -len(finding.classifications),
                finding.document_filename.lower(),
            )
        )
        return findings

    def _duplication_signals(
        self,
        findings: list[DuplicationFinding],
        document_lookup: dict[str, Document],
    ) -> tuple[set[str], dict[str, dict[str, int]]]:
        """Map duplication findings back to document-level counts."""
        duplicated_chunk_ids: set[str] = set()
        stats_by_document: dict[str, dict[str, int]] = {}

        for document_id in document_lookup:
            stats_by_document[document_id] = {
                "exact": 0,
                "near_duplicate": 0,
                "already_in_index": 0,
            }

        for finding in findings:
            for chunk in finding.chunks_involved:
                if chunk.source_scope != "new":
                    continue
                if chunk.parent_document_id not in document_lookup:
                    continue
                duplicated_chunk_ids.add(chunk.chunk_id)
                stats_by_document[chunk.parent_document_id][finding.finding_type] += 1

        return duplicated_chunk_ids, stats_by_document

    def _staleness_signals(self, findings: list[StalenessFinding]) -> dict[str, list[StalenessFinding]]:
        """Group staleness findings by current document identifier."""
        grouped: dict[str, list[StalenessFinding]] = {}
        for finding in findings:
            document_id = finding.chunk.parent_document_id
            if not document_id:
                continue
            grouped.setdefault(document_id, []).append(finding)
        return grouped

    def _supersession_signals(self, findings: list[StalenessFinding]) -> dict[str, list[StalenessFinding]]:
        """Group supersession findings by document identifier when available."""
        grouped: dict[str, list[StalenessFinding]] = {}
        for finding in findings:
            document_id = finding.chunk.parent_document_id
            if not document_id:
                continue
            grouped.setdefault(document_id, []).append(finding)
        return grouped

    def _contradiction_signals(
        self,
        findings: list[ContradictionFinding],
    ) -> dict[str, list[ContradictionFinding]]:
        """Group contradiction findings by document identifier."""
        grouped: dict[str, list[ContradictionFinding]] = {}
        for finding in findings:
            seen_document_ids: set[str] = set()
            for chunk in finding.chunks_involved:
                if not chunk.parent_document_id or chunk.parent_document_id in seen_document_ids:
                    continue
                seen_document_ids.add(chunk.parent_document_id)
                grouped.setdefault(chunk.parent_document_id, []).append(finding)
        return grouped

    async def _assess_triviality(
        self,
        document_chunks: list[Chunk],
        *,
        api_key: str | None,
        llm_checks_remaining: int,
    ) -> TrivialityDocumentAssessment:
        """Score document information density using chunk-level heuristics and optional LLM checks."""
        if not document_chunks:
            return TrivialityDocumentAssessment(
                is_trivial=False,
                average_word_count=0.0,
                average_unique_ratio=0.0,
                average_boilerplate_ratio=0.0,
                trivial_chunk_ratio=0.0,
                boilerplate_patterns=[],
                llm_checks_used=0,
                llm_supported_triviality=False,
            )

        chunk_assessments = [self._assess_chunk(chunk) for chunk in document_chunks]
        average_word_count = sum(item.word_count for item in chunk_assessments) / len(chunk_assessments)
        average_unique_ratio = sum(item.unique_ratio for item in chunk_assessments) / len(chunk_assessments)
        average_boilerplate_ratio = sum(item.boilerplate_ratio for item in chunk_assessments) / len(chunk_assessments)
        trivial_chunk_ratio = sum(1 for item in chunk_assessments if item.is_trivial) / len(chunk_assessments)

        patterns = sorted(
            {
                hit.lower()
                for item in chunk_assessments
                for hit in item.boilerplate_hits
            }
        )

        heuristically_trivial = (
            (average_word_count < 30 and average_unique_ratio < 0.4)
            or trivial_chunk_ratio >= 0.5
            or average_boilerplate_ratio > 0.04
        )
        heuristically_healthy = (
            average_word_count >= 45
            and average_unique_ratio >= 0.45
            and average_boilerplate_ratio < 0.02
            and trivial_chunk_ratio < 0.34
        )

        llm_checks_used = 0
        llm_supported_triviality = False
        if not heuristically_trivial and not heuristically_healthy and api_key and llm_checks_remaining > 0:
            borderline_chunks = [item for item in chunk_assessments if item.is_borderline][:llm_checks_remaining]
            for item in borderline_chunks:
                verdict = await self._llm_chunk_substance(item.chunk.text, api_key)
                item.llm_verdict = verdict
                llm_checks_used += 1
            non_substantive = sum(1 for item in borderline_chunks if item.llm_verdict == "TRIVIAL")
            llm_supported_triviality = bool(borderline_chunks) and non_substantive >= max(1, len(borderline_chunks) // 2)
            if llm_supported_triviality:
                heuristically_trivial = True

        return TrivialityDocumentAssessment(
            is_trivial=heuristically_trivial,
            average_word_count=average_word_count,
            average_unique_ratio=average_unique_ratio,
            average_boilerplate_ratio=average_boilerplate_ratio,
            trivial_chunk_ratio=trivial_chunk_ratio,
            boilerplate_patterns=patterns,
            llm_checks_used=llm_checks_used,
            llm_supported_triviality=llm_supported_triviality,
        )

    def _assess_chunk(self, chunk: Chunk) -> TrivialityChunkAssessment:
        """Extract heuristic information-density signals from a chunk."""
        words = [match.group(0).lower() for match in self.WORD_PATTERN.finditer(chunk.text)]
        word_count = len(words)
        unique_ratio = (len(set(words)) / word_count) if word_count else 0.0

        lower_text = chunk.text.lower()
        boilerplate_hits = [
            pattern
            for pattern in self.BOILERPLATE_PATTERNS
            if pattern.lower() in lower_text
        ]
        boilerplate_ratio = len(boilerplate_hits) / max(word_count, 1)

        is_trivial = (
            (word_count < 20 and unique_ratio < 0.5)
            or (word_count < 30 and unique_ratio < 0.4)
            or unique_ratio < 0.3
            or bool(boilerplate_hits and word_count < 80)
        )
        is_borderline = (
            not is_trivial
            and (
                word_count < 40
                or 0.35 <= unique_ratio < 0.45
                or bool(boilerplate_hits)
            )
        )

        return TrivialityChunkAssessment(
            chunk=chunk,
            word_count=word_count,
            unique_ratio=round(unique_ratio, 3),
            boilerplate_hits=boilerplate_hits,
            boilerplate_ratio=round(boilerplate_ratio, 3),
            is_trivial=is_trivial,
            is_borderline=is_borderline,
        )

    async def _llm_chunk_substance(self, text: str, api_key: str) -> str:
        """Use a minimal chat completion to break a borderline triviality tie."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("The openai package is required for ROT LLM checks.") from exc

        client = AsyncOpenAI(api_key=api_key.strip())
        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                temperature=0,
                max_completion_tokens=100,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.SUBSTANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("ROT triviality check failed: %s", exc)
            return "SUBSTANTIVE"

        content = self._extract_content(response)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return "SUBSTANTIVE"

        status = str(payload.get("status", "SUBSTANTIVE")).upper()
        return "TRIVIAL" if status == "TRIVIAL" else "SUBSTANTIVE"

    def _extract_content(self, response: Any) -> str:
        """Extract message content from a chat completions response."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        return str(content or "")

    def _build_explanation(
        self,
        filename: str,
        classifications: list[str],
        *,
        duplicate_ratio: float,
        average_staleness: float,
        triviality: TrivialityDocumentAssessment,
    ) -> str:
        """Explain why a document received its ROT classification."""
        if classifications == ["healthy"]:
            return f"{filename} does not show strong redundant, outdated, or trivial signals in the current analysis."

        reasons: list[str] = []
        if "redundant" in classifications:
            reasons.append(f"{duplicate_ratio:.0%} of its chunks overlap with duplicate content")
        if "outdated" in classifications:
            reasons.append(f"its average staleness score is {average_staleness:.2f}")
        if "trivial" in classifications:
            reasons.append(
                "its chunks are short or low-density"
                f" with an average of {triviality.average_word_count:.1f} words and a unique-term ratio of {triviality.average_unique_ratio:.2f}"
            )
        return f"{filename} was flagged because " + ", ".join(reasons) + "."

    def _build_impact(self, classifications: list[str], contradiction_count: int) -> str:
        """Describe the likely RAG impact of the ROT signals."""
        if classifications == ["healthy"]:
            impact = "This document is unlikely to harm retrieval quality on its own."
        else:
            parts: list[str] = []
            if "redundant" in classifications:
                parts.append("it can crowd better sources out of retrieval")
            if "outdated" in classifications:
                parts.append("it can cause the model to cite stale facts")
            if "trivial" in classifications:
                parts.append("it can waste context window on low-value text")
            impact = "If indexed as-is, " + ", ".join(parts) + "."
        if contradiction_count:
            impact += f" This document also appears in {contradiction_count} contradiction finding(s), which raises answer consistency risk."
        return impact

    def _build_recommendation(self, classifications: list[str]) -> str:
        """Recommend the next document-level action."""
        if classifications == ["healthy"]:
            return "Review complete. Keep this document in the corpus."
        if "redundant" in classifications and "outdated" in classifications:
            return "Remove this document if a fresher canonical source already exists. Otherwise update it and keep only one version."
        if "redundant" in classifications:
            return "Remove or merge this document so only one canonical version remains in the corpus."
        if "outdated" in classifications:
            return "Update this document or replace it with a newer source before the next indexing run."
        return "Review this document and remove it if it is only boilerplate, placeholder text, or non-retrievable material."
