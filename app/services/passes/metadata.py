"""Metadata audit pass for Phase 3."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.models.document import Document, MetadataAuditSummary, MetadataFinding

DEFAULT_METADATA_FIELD_INPUT = "title, author, date modified"


def parse_metadata_field_input(raw_value: str | None) -> list[str] | None:
    """Parse a comma or newline separated metadata field list from the upload form."""
    if raw_value is None:
        return None

    candidates = [
        item.strip()
        for item in re.split(r"[\r\n,]+", raw_value)
        if item.strip()
    ]
    return candidates or None


@dataclass(slots=True)
class DocumentMetadataAssessment:
    """Internal metadata audit result for a document."""

    document: Document
    completeness_score: float
    missing_fields: list[str]
    raw_date_formats: set[str]
    has_any_dates: bool


@dataclass(slots=True)
class ExpectedMetadataField:
    """A configured metadata field to audit."""

    key: str
    label: str
    lookup_keys: set[str]


class MetadataAuditPass:
    """Audit document metadata completeness and corpus consistency."""

    DEFAULT_EXPECTED_FIELDS = (
        "title",
        "author_or_owner",
        "created_date",
        "modified_date",
        "version_information",
        "document_type_or_category",
    )
    FIELD_ALIASES = {
        "title": "title",
        "document title": "title",
        "title or meaningful filename": "title",
        "author": "author_or_owner",
        "owner": "author_or_owner",
        "author or owner": "author_or_owner",
        "created": "created_date",
        "created at": "created_date",
        "created date": "created_date",
        "date created": "created_date",
        "modified": "modified_date",
        "modified at": "modified_date",
        "modified date": "modified_date",
        "date modified": "modified_date",
        "last modified": "modified_date",
        "updated": "modified_date",
        "version": "version_information",
        "revision": "version_information",
        "version information": "version_information",
        "product version": "version_information",
        "category": "document_type_or_category",
        "type": "document_type_or_category",
        "document type": "document_type_or_category",
        "document category": "document_type_or_category",
        "document type or category": "document_type_or_category",
    }
    GENERIC_FILENAME_PATTERN = re.compile(
        r"^(?:doc(?:ument)?(?:[\s_-]?\d+)?|file(?:[\s_-]?\d+)?|untitled(?:[\s_-]?\d+)?|new document(?:[\s_-]?\d+)?|scan(?:[\s_-]?\d+)?)$",
        re.IGNORECASE,
    )
    ISO_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T")
    ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    MONTH_NAME_PATTERN = re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        re.IGNORECASE,
    )
    SLASH_DATE_PATTERN = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")

    def __init__(self, *, expected_fields: list[str] | None = None) -> None:
        self.expected_fields = self._resolve_expected_fields(expected_fields)

    def audit_documents(
        self,
        documents: list[Document],
    ) -> tuple[list[MetadataFinding], MetadataAuditSummary]:
        """Return per-document metadata findings and a corpus-level summary."""
        if not documents:
            return [], MetadataAuditSummary(
                average_completeness=0.0,
                expected_fields=[field.label for field in self.expected_fields],
            )

        assessments = [self._assess_document(document) for document in documents]

        format_labels = Counter()
        missing_counter = Counter()
        documents_with_dates = 0
        documents_without_dates = 0

        for assessment in assessments:
            missing_counter.update(assessment.missing_fields)
            format_labels.update(assessment.raw_date_formats)
            if assessment.has_any_dates:
                documents_with_dates += 1
            else:
                documents_without_dates += 1

        consistency_issues: list[str] = []
        mixed_date_formats = len(format_labels) > 1
        if mixed_date_formats:
            consistency_issues.append("Date metadata appears in multiple formats across the corpus.")
        if documents_with_dates and documents_without_dates:
            consistency_issues.append(
                "Some documents have created or modified dates while others have no date metadata at all."
            )

        findings: list[MetadataFinding] = []
        for assessment in assessments:
            document_consistency_issues: list[str] = []
            if mixed_date_formats and assessment.raw_date_formats:
                document_consistency_issues.append("This document uses a date format that is not consistent across the corpus.")
            if not assessment.has_any_dates and documents_with_dates:
                document_consistency_issues.append("This document has no created or modified date while other documents do.")

            findings.append(
                MetadataFinding(
                    document_id=assessment.document.id,
                    document_filename=assessment.document.metadata.filename,
                    completeness_score=assessment.completeness_score,
                    missing_fields=assessment.missing_fields,
                    consistency_issues=document_consistency_issues,
                    explanation=self._build_explanation(assessment.completeness_score, assessment.missing_fields),
                    recommendation=self._build_recommendation(
                        assessment.completeness_score,
                        assessment.missing_fields,
                        document_consistency_issues,
                    ),
                )
            )

        findings.sort(key=lambda finding: finding.completeness_score)
        summary = MetadataAuditSummary(
            average_completeness=round(
                sum(assessment.completeness_score for assessment in assessments) / len(assessments),
                3,
            ),
            most_common_missing_fields=dict(missing_counter.most_common()),
            consistency_issues=consistency_issues,
            expected_fields=[field.label for field in self.expected_fields],
        )

        return findings, summary

    def score_document_metadata(self, document: Document) -> float:
        """Return only the completeness score for a document."""
        return self._assess_document(document).completeness_score

    def _assess_document(self, document: Document) -> DocumentMetadataAssessment:
        """Evaluate a document against the expected metadata fields."""
        embedded = document.metadata.embedded
        frontmatter = document.metadata.frontmatter

        canonical_presence = {
            "title": self._has_title(document),
            "author_or_owner": self._has_value(
                embedded.get("author"),
                embedded.get("last_modified_by"),
                frontmatter.get("author"),
                frontmatter.get("owner"),
            ),
            "created_date": self._has_value(
                document.metadata.created_at,
                embedded.get("created"),
                frontmatter.get("created"),
                frontmatter.get("date"),
            ),
            "modified_date": self._has_value(
                document.metadata.modified_at,
                embedded.get("modified"),
                frontmatter.get("modified"),
                frontmatter.get("updated"),
                frontmatter.get("last_modified"),
            ),
            "version_information": self._has_value(
                embedded.get("version"),
                embedded.get("revision"),
                frontmatter.get("version"),
            ),
            "document_type_or_category": self._has_value(
                embedded.get("category"),
                frontmatter.get("category"),
                frontmatter.get("type"),
                frontmatter.get("document_type"),
            ),
        }

        missing_fields: list[str] = []
        present_count = 0
        for field in self.expected_fields:
            if self._field_is_present(document, field, canonical_presence):
                present_count += 1
            else:
                missing_fields.append(field.label)
        completeness_score = round(present_count / len(self.expected_fields), 3)

        return DocumentMetadataAssessment(
            document=document,
            completeness_score=completeness_score,
            missing_fields=missing_fields,
            raw_date_formats=self._collect_date_formats(document),
            has_any_dates=canonical_presence["created_date"] or canonical_presence["modified_date"],
        )

    def _has_title(self, document: Document) -> bool:
        """Return whether the document has a meaningful title or filename."""
        embedded_title = document.metadata.embedded.get("title")
        frontmatter_title = document.metadata.frontmatter.get("title")
        if self._has_value(embedded_title, frontmatter_title):
            return True

        filename_stem = Path(document.metadata.filename).stem.strip()
        if not filename_stem:
            return False
        return not self.GENERIC_FILENAME_PATTERN.match(filename_stem)

    def _collect_date_formats(self, document: Document) -> set[str]:
        """Classify the date formats present in document metadata."""
        formats: set[str] = set()
        if document.metadata.created_at or document.metadata.modified_at:
            formats.add("normalized_datetime")

        for key, value in {
            **document.metadata.embedded,
            **document.metadata.frontmatter,
        }.items():
            if not any(token in key.lower() for token in ("date", "created", "modified", "updated")):
                continue
            format_label = self._classify_date_value(value)
            if format_label:
                formats.add(format_label)

        return formats

    def _classify_date_value(self, value: Any) -> str | None:
        """Map a metadata value to a rough date-format label."""
        if isinstance(value, datetime):
            return "datetime_object"
        if isinstance(value, date):
            return "date_object"
        if not isinstance(value, str):
            return None

        candidate = value.strip()
        if not candidate:
            return None
        if self.ISO_DATETIME_PATTERN.match(candidate):
            return "iso_datetime"
        if self.ISO_DATE_PATTERN.match(candidate):
            return "iso_date"
        if self.SLASH_DATE_PATTERN.match(candidate):
            return "slash_date"
        if self.MONTH_NAME_PATTERN.match(candidate):
            return "month_name"
        return "other_date_string"

    def _has_value(self, *values: Any) -> bool:
        """Return whether any candidate metadata value is meaningfully populated."""
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return True
        return False

    def _field_label(self, field_name: str) -> str:
        """Convert internal field names into display labels."""
        labels = {
            "title": "title or meaningful filename",
            "author_or_owner": "author or owner",
            "created_date": "created date",
            "modified_date": "modified date",
            "version_information": "version information",
            "document_type_or_category": "document type or category",
        }
        return labels[field_name]

    def _field_is_present(
        self,
        document: Document,
        field: ExpectedMetadataField,
        canonical_presence: dict[str, bool],
    ) -> bool:
        """Check whether a configured field is populated on a document."""
        if field.key in canonical_presence:
            return canonical_presence[field.key]
        return self._has_custom_metadata_value(document, field.lookup_keys)

    def _has_custom_metadata_value(self, document: Document, lookup_keys: set[str]) -> bool:
        """Check document metadata dictionaries for a custom configured field."""
        metadata_values: dict[str, Any] = {
            self._normalize_field_name(key): value
            for key, value in {
                **document.metadata.embedded,
                **document.metadata.frontmatter,
            }.items()
        }
        metadata_values.update(
            {
                "filename": document.metadata.filename,
                "extension": document.metadata.extension,
                "file size": document.metadata.file_size,
                "created date": document.metadata.created_at,
                "modified date": document.metadata.modified_at,
            }
        )
        return any(self._has_value(metadata_values.get(key)) for key in lookup_keys)

    def _resolve_expected_fields(self, expected_fields: list[str] | None) -> list[ExpectedMetadataField]:
        """Normalize requested metadata fields or fall back to the standard audit set."""
        requested = [
            field.strip()
            for field in (expected_fields or [])
            if isinstance(field, str) and field.strip()
        ]
        if not requested:
            requested = [self._field_label(field_name) for field_name in self.DEFAULT_EXPECTED_FIELDS]

        resolved: list[ExpectedMetadataField] = []
        seen: set[str] = set()
        for requested_field in requested:
            normalized = self._normalize_field_name(requested_field)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            canonical_key = self.FIELD_ALIASES.get(normalized)
            lookup_keys = (
                {canonical_key}
                if canonical_key
                else {
                    normalized,
                    normalized.replace(" ", ""),
                }
            )
            resolved.append(
                ExpectedMetadataField(
                    key=canonical_key or normalized,
                    label=requested_field,
                    lookup_keys=lookup_keys,
                )
            )

        return resolved or [
            ExpectedMetadataField(
                key=field_name,
                label=self._field_label(field_name),
                lookup_keys={field_name},
            )
            for field_name in self.DEFAULT_EXPECTED_FIELDS
        ]

    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize a user-provided field name for matching."""
        return " ".join(
            field_name.strip().lower().replace("_", " ").replace("-", " ").split()
        )

    def _build_explanation(self, score: float, missing_fields: list[str]) -> str:
        """Explain why the document metadata score matters."""
        if score < 0.3:
            return (
                "This document is missing most of the metadata needed to rank, filter, and trust it during retrieval."
            )
        if score < 0.6:
            return (
                "This document has partial metadata coverage, which may limit retrieval quality and make review harder."
            )
        if missing_fields:
            return (
                "This document has usable metadata overall, but a few missing fields still reduce ranking and governance options."
            )
        return "This document has strong metadata coverage for downstream retrieval and audit workflows."

    def _build_recommendation(
        self,
        score: float,
        missing_fields: list[str],
        consistency_issues: list[str],
    ) -> str:
        """Recommend the next metadata action for a document."""
        if score < 0.3:
            return "Add ownership, dating, and classification metadata before relying on this document in retrieval."
        if score < 0.6:
            return "Fill in the missing fields and standardize the metadata format before the next indexing run."
        if consistency_issues:
            return "Keep the metadata, but standardize the inconsistent fields so the corpus follows one format."
        if missing_fields:
            return "Add the remaining missing metadata fields when the document is next updated."
        return "Keep this metadata as the baseline for future documents in the corpus."
