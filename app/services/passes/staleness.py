"""Staleness scoring pass for Phase 3."""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.models.document import Chunk, Document, FindingChunk, StalenessFinding
from app.services.similarity import SimilarityPair


MONTH_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)


@dataclass(slots=True)
class ChunkAssessment:
    """Internal staleness assessment state for a chunk."""

    chunk: Chunk
    score: float
    signals: dict[str, object]
    reference_date: datetime | None


class StalenessScoringPass:
    """Score chunk freshness using metadata and text-based temporal cues."""

    ISO_DATE_PATTERN = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
    QUARTER_PATTERN = re.compile(r"\bQ(?P<quarter>[1-4])\s*(?P<year>\d{4})\b", re.IGNORECASE)
    FISCAL_YEAR_PATTERN = re.compile(r"\bFY\s*(?P<year>\d{4})\b", re.IGNORECASE)
    MONTH_YEAR_PATTERN = re.compile(
        rf"\b(?:as of\s+)?(?P<month>{MONTH_PATTERN})\s+(?:(?P<day>\d{{1,2}}),\s+)?(?P<year>\d{{4}})\b",
        re.IGNORECASE,
    )
    VERSION_PATTERN = re.compile(r"\b(?:version|v)\s*\d+(?:\.\d+)+\b", re.IGNORECASE)
    TEMPORAL_LANGUAGE_PATTERN = re.compile(
        r"\b(currently|this year|recently|at this time|as of now|today|current)\b",
        re.IGNORECASE,
    )
    CHUNK_SCOPE_THRESHOLD = 0.7
    MIN_REPORTING_SCORE = 0.2

    def __init__(
        self,
        *,
        stale_after_months: int = 12,
        metadata_weight: float = 0.6,
        content_weight: float = 0.4,
        current_datetime: datetime | None = None,
    ) -> None:
        if stale_after_months <= 0:
            raise ValueError("stale_after_months must be positive")
        self.stale_after_months = stale_after_months
        self.metadata_weight = metadata_weight
        self.content_weight = content_weight
        self.current_datetime = current_datetime or datetime.now(UTC)

    def build_findings(
        self,
        new_chunks: list[Chunk],
        documents: list[Document],
        *,
        existing_chunks: list[Chunk] | None = None,
        similarity_pairs: list[SimilarityPair] | None = None,
    ) -> tuple[list[StalenessFinding], list[StalenessFinding]]:
        """Build per-chunk staleness findings and brownfield supersession findings."""
        document_lookup = {document.id: document for document in documents}
        assessments: dict[str, ChunkAssessment] = {}

        for chunk in new_chunks:
            assessments[chunk.id] = self._assess_chunk(chunk, document_lookup)

        for chunk in existing_chunks or []:
            assessments[chunk.id] = self._assess_chunk(chunk, document_lookup)

        chunk_findings = [
            self._standard_finding(assessments[chunk.id], source_scope="new")
            for chunk in new_chunks
            if assessments[chunk.id].score > self.MIN_REPORTING_SCORE
        ]
        chunk_findings.sort(key=lambda finding: finding.staleness_score, reverse=True)

        supersession_findings = self._build_supersession_findings(
            assessments,
            similarity_pairs or [],
        )

        return chunk_findings, supersession_findings

    def extract_content_signals(self, text: str) -> dict[str, object]:
        """Extract date references and temporal phrases from chunk text."""
        detected_dates: list[dict[str, object]] = []
        seen_dates: set[tuple[str, str]] = set()

        for match in self.ISO_DATE_PATTERN.finditer(text):
            normalized = self._build_date(
                year=int(match.group("year")),
                month=int(match.group("month")),
                day=int(match.group("day")),
            )
            self._append_detected_date(detected_dates, seen_dates, match.group(0), normalized)

        for match in self.QUARTER_PATTERN.finditer(text):
            quarter = int(match.group("quarter"))
            year = int(match.group("year"))
            normalized = self._build_date(year=year, month=quarter * 3, day=1)
            self._append_detected_date(detected_dates, seen_dates, match.group(0), normalized)

        for match in self.FISCAL_YEAR_PATTERN.finditer(text):
            year = int(match.group("year"))
            normalized = self._build_date(year=year, month=12, day=1)
            self._append_detected_date(detected_dates, seen_dates, match.group(0), normalized)

        for match in self.MONTH_YEAR_PATTERN.finditer(text):
            month = self._month_number(match.group("month"))
            year = int(match.group("year"))
            day_value = match.group("day")
            day = int(day_value) if day_value else 1
            normalized = self._build_date(year=year, month=month, day=day)
            self._append_detected_date(detected_dates, seen_dates, match.group(0), normalized)

        temporal_language = sorted(
            {
                match.group(0).lower()
                for match in self.TEMPORAL_LANGUAGE_PATTERN.finditer(text)
            }
        )
        version_references = sorted(
            {
                match.group(0).lower()
                for match in self.VERSION_PATTERN.finditer(text)
            }
        )

        explicit_reference_date = None
        content_age_months = None
        content_score = None
        content_confidence = None
        stale_reference_count = 0
        stale_reference_ratio = None

        if detected_dates:
            explicit_reference_date = max(
                datetime.combine(item["normalized_date"], datetime.min.time(), tzinfo=UTC)
                for item in detected_dates
            )
            content_age_months = self._months_between(explicit_reference_date, self.current_datetime)
            content_score = self._age_score(content_age_months)
            date_scores = [
                self._age_score(
                    self._months_between(
                        datetime.combine(item["normalized_date"], datetime.min.time(), tzinfo=UTC),
                        self.current_datetime,
                    )
                )
                for item in detected_dates
            ]
            stale_reference_count = sum(1 for score in date_scores if score >= 0.7)
            stale_reference_ratio = stale_reference_count / len(date_scores)
            if stale_reference_ratio >= 0.75 and content_score >= 0.7:
                content_score = min(1.0, content_score + min(0.2, 0.08 * max(len(detected_dates) - 1, 0)))
            if temporal_language and content_score >= 0.5:
                content_score = min(1.0, content_score + 0.1)
            if version_references and content_score >= 0.5:
                content_score = min(1.0, content_score + 0.05)
            content_confidence = self._content_confidence(
                detected_date_count=len(detected_dates),
                stale_reference_ratio=stale_reference_ratio,
                has_temporal_language=bool(temporal_language),
                has_version_references=bool(version_references),
            )
        elif temporal_language and version_references:
            content_score = 0.65
            content_confidence = 0.6
        elif temporal_language:
            content_score = 0.6
            content_confidence = 0.55
        elif version_references:
            content_score = 0.4
            content_confidence = 0.5

        return {
            "detected_dates": [item["raw_text"] for item in detected_dates],
            "detected_date_values": [item["normalized_date"].isoformat() for item in detected_dates],
            "detected_date_count": len(detected_dates),
            "stale_reference_count": stale_reference_count,
            "stale_reference_ratio": round(stale_reference_ratio, 3) if stale_reference_ratio is not None else None,
            "temporal_language": temporal_language,
            "version_references": version_references,
            "content_reference_date": explicit_reference_date.isoformat() if explicit_reference_date else None,
            "content_age_months": content_age_months,
            "content_score": round(content_score, 3) if content_score is not None else None,
            "content_confidence": round(content_confidence, 3) if content_confidence is not None else None,
        }

    def _assess_chunk(
        self,
        chunk: Chunk,
        document_lookup: dict[str, Document],
    ) -> ChunkAssessment:
        """Combine metadata and content freshness signals into a single score."""
        metadata_dates = self._metadata_dates_for_chunk(chunk, document_lookup)
        metadata_reference_date = None
        metadata_age_months = None
        metadata_score = None

        candidate_metadata_dates = [value for value in metadata_dates.values() if value is not None]
        if candidate_metadata_dates:
            metadata_reference_date = max(candidate_metadata_dates)
            metadata_age_months = self._months_between(metadata_reference_date, self.current_datetime)
            metadata_score = self._age_score(metadata_age_months)

        content_signals = self.extract_content_signals(chunk.text)
        content_score = content_signals["content_score"]
        content_confidence = content_signals["content_confidence"]
        content_reference_date = self._parse_iso_datetime(content_signals["content_reference_date"])

        if metadata_score is not None and content_score is not None:
            weighted_score = (self.metadata_weight * metadata_score) + (self.content_weight * content_score)
            content_floor = float(content_score) * float(content_confidence or 0.0)
            score = max(weighted_score, content_floor)
        elif metadata_score is not None:
            score = metadata_score
        elif content_score is not None:
            score = float(content_score)
        else:
            score = 0.5

        reference_date = metadata_reference_date or content_reference_date
        signals = {
            "metadata_dates": {
                key: value.isoformat() if value else None
                for key, value in metadata_dates.items()
            },
            "metadata_reference_date": metadata_reference_date.isoformat() if metadata_reference_date else None,
            "metadata_age_months": metadata_age_months,
            "metadata_score": round(metadata_score, 3) if metadata_score is not None else None,
            **content_signals,
        }

        return ChunkAssessment(
            chunk=chunk,
            score=round(score, 3),
            signals=signals,
            reference_date=reference_date,
        )

    def _standard_finding(self, assessment: ChunkAssessment, *, source_scope: str) -> StalenessFinding:
        """Build the standard per-chunk staleness finding."""
        score = assessment.score
        has_any_dates = bool(
            assessment.signals["metadata_reference_date"] or assessment.signals["content_reference_date"]
        )

        if not has_any_dates and not assessment.signals["temporal_language"] and not assessment.signals["version_references"]:
            explanation = (
                "This chunk has no usable freshness signals in metadata or text. That makes it harder to know "
                "whether the content is still safe to retrieve."
            )
            recommendation = "Review this content and add clearer dates or version metadata."
        elif score > 0.7:
            explanation = (
                "This chunk looks strongly anchored to older dates or stale metadata and should be treated as likely outdated."
            )
            recommendation = "Review or remove before indexing."
        elif score >= 0.4:
            explanation = (
                "This chunk may be outdated based on the age and concentration of the freshness signals that were detected."
            )
            recommendation = "May be outdated, review recommended."
        else:
            explanation = (
                "This chunk appears relatively current based on the metadata and temporal cues that were detected."
            )
            recommendation = "Keep, but monitor."

        return StalenessFinding(
            finding_type="staleness",
            chunk=self._snapshot(assessment.chunk, source_scope=source_scope),
            staleness_score=assessment.score,
            signals=assessment.signals,
            explanation=explanation,
            recommendation=recommendation,
        )

    def _build_supersession_findings(
        self,
        assessments: dict[str, ChunkAssessment],
        similarity_pairs: list[SimilarityPair],
    ) -> list[StalenessFinding]:
        """Flag existing chunks that appear older than similar new content."""
        supersession_findings: list[StalenessFinding] = []
        seen_pairs: set[tuple[str, str]] = set()

        for pair in similarity_pairs:
            if pair.comparison_scope != "new_vs_existing":
                continue
            if pair.similarity_score < self.CHUNK_SCOPE_THRESHOLD:
                continue

            new_assessment = assessments.get(pair.left_chunk.id)
            existing_assessment = assessments.get(pair.right_chunk.id)
            if new_assessment is None or existing_assessment is None:
                continue
            if existing_assessment.score <= self.MIN_REPORTING_SCORE:
                continue
            if new_assessment.reference_date is None or existing_assessment.reference_date is None:
                continue
            if new_assessment.reference_date <= existing_assessment.reference_date + timedelta(days=1):
                continue

            pair_key = (pair.left_chunk.id, pair.right_chunk.id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            signals = dict(existing_assessment.signals)
            signals["matched_newer_chunk_filename"] = pair.left_chunk.parent_document_name or "Unknown source"
            signals["matched_newer_reference_date"] = new_assessment.reference_date.isoformat()
            signals["matched_similarity_score"] = round(pair.similarity_score, 3)

            supersession_findings.append(
                StalenessFinding(
                    finding_type="potentially_superseded",
                    chunk=self._snapshot(existing_assessment.chunk, source_scope="existing"),
                    related_chunk=self._snapshot(new_assessment.chunk, source_scope="new"),
                    staleness_score=existing_assessment.score,
                    signals=signals,
                    explanation=(
                        "This existing chunk matches a newer uploaded chunk on the same topic. Keeping the older "
                        "indexed version may cause retrieval to surface outdated content."
                    ),
                    recommendation=(
                        "Review the existing indexed chunk and replace or demote it if the newer uploaded content "
                        "is the current source of truth."
                    ),
                )
            )

        supersession_findings.sort(key=lambda finding: finding.staleness_score, reverse=True)
        return supersession_findings

    def _metadata_dates_for_chunk(
        self,
        chunk: Chunk,
        document_lookup: dict[str, Document],
    ) -> dict[str, datetime | None]:
        """Collect metadata dates from a document-backed chunk or an imported chunk."""
        created_at = None
        modified_at = None

        document = document_lookup.get(chunk.parent_document_id or "")
        if document is not None:
            created_at = self._ensure_utc(document.metadata.created_at)
            modified_at = self._ensure_utc(document.metadata.modified_at)
        else:
            created_at = self._coerce_datetime(
                chunk.metadata.get("created_at")
                or chunk.metadata.get("created")
                or chunk.metadata.get("date_created")
            )
            modified_at = self._coerce_datetime(
                chunk.metadata.get("modified_at")
                or chunk.metadata.get("modified")
                or chunk.metadata.get("updated_at")
                or chunk.metadata.get("updated")
                or chunk.metadata.get("last_modified")
            )

        return {
            "created_at": created_at,
            "modified_at": modified_at,
        }

    def _snapshot(self, chunk: Chunk, *, source_scope: str) -> FindingChunk:
        """Build a finding chunk snapshot for the report."""
        return FindingChunk(
            chunk_id=chunk.id,
            text=chunk.text,
            source_filename=chunk.parent_document_name or "Unknown source",
            parent_document_id=chunk.parent_document_id,
            source_scope=source_scope,
        )

    def _append_detected_date(
        self,
        detected_dates: list[dict[str, object]],
        seen_dates: set[tuple[str, str]],
        raw_text: str,
        normalized_date: date | None,
    ) -> None:
        """Store a parsed date match if it is valid and not already seen."""
        if normalized_date is None:
            return
        key = (raw_text.lower(), normalized_date.isoformat())
        if key in seen_dates:
            return
        seen_dates.add(key)
        detected_dates.append(
            {
                "raw_text": raw_text,
                "normalized_date": normalized_date,
            }
        )

    def _build_date(self, *, year: int, month: int, day: int) -> date | None:
        """Create a valid date from extracted components."""
        try:
            max_day = calendar.monthrange(year, month)[1]
            return date(year, month, min(day, max_day))
        except ValueError:
            return None

    def _month_number(self, month_name: str) -> int:
        """Convert a month name or abbreviation into its numeric month."""
        normalized = month_name.strip().lower()[:3]
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        return month_map[normalized]

    def _months_between(self, older: datetime, newer: datetime) -> int:
        """Return the number of whole months between two datetimes."""
        older_utc = self._ensure_utc(older)
        newer_utc = self._ensure_utc(newer)
        if older_utc is None or newer_utc is None:
            return 0
        if older_utc > newer_utc:
            return 0

        months = (newer_utc.year - older_utc.year) * 12 + (newer_utc.month - older_utc.month)
        if newer_utc.day < older_utc.day:
            months -= 1
        return max(months, 0)

    def _age_score(self, age_months: int) -> float:
        """Convert an age in months into a steeper staleness score."""
        if age_months <= 0:
            return 0.0
        score = 1 - math.exp(-(age_months / self.stale_after_months))
        return round(min(score, 1.0), 3)

    def _content_confidence(
        self,
        *,
        detected_date_count: int,
        stale_reference_ratio: float,
        has_temporal_language: bool,
        has_version_references: bool,
    ) -> float:
        """Estimate confidence that explicit text signals reflect genuinely stale content."""
        confidence = 0.55
        confidence += min(0.24, 0.14 * max(detected_date_count - 1, 0))
        if stale_reference_ratio >= 0.75:
            confidence += 0.12
        if has_temporal_language:
            confidence += 0.05
        if has_version_references:
            confidence += 0.03
        return round(min(confidence, 0.95), 3)

    def _coerce_datetime(self, value: Any) -> datetime | None:
        """Parse a datetime-like metadata value into UTC."""
        if isinstance(value, datetime):
            return self._ensure_utc(value)
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
        if not isinstance(value, str):
            return None

        candidate = value.strip()
        if not candidate:
            return None

        for attempt in (
            candidate,
            candidate.replace("Z", "+00:00"),
        ):
            try:
                parsed = datetime.fromisoformat(attempt)
            except ValueError:
                continue
            return self._ensure_utc(parsed)

        for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                parsed_date = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return parsed_date.replace(tzinfo=UTC)

        return None

    def _parse_iso_datetime(self, value: object) -> datetime | None:
        """Parse a previously serialized ISO datetime string."""
        if not isinstance(value, str) or not value:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime | None) -> datetime | None:
        """Normalize datetimes to UTC for stable comparisons."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
