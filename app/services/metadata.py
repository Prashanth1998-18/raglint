"""Helpers for extracting and normalizing metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from yaml import YAMLError

from app.models.document import DocumentMetadata


def build_document_metadata(
    *,
    filename: str,
    file_size: int,
    created_at: datetime | None = None,
    modified_at: datetime | None = None,
    embedded: dict[str, object] | None = None,
    frontmatter: dict[str, object] | None = None,
) -> DocumentMetadata:
    """Build a document metadata model with normalized fields."""
    return DocumentMetadata(
        filename=filename,
        extension=Path(filename).suffix.lower(),
        file_size=file_size,
        created_at=created_at,
        modified_at=modified_at,
        embedded=embedded or {},
        frontmatter=frontmatter or {},
    )


def parse_markdown_frontmatter(raw_text: str) -> tuple[dict[str, object], str]:
    """Parse YAML frontmatter from a markdown document when present."""
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    end_marker = raw_text.find("\n---\n", 4)
    if end_marker == -1:
        return {}, raw_text

    frontmatter_block = raw_text[4:end_marker]
    body = raw_text[end_marker + len("\n---\n") :]
    try:
        parsed = yaml.safe_load(frontmatter_block) or {}
    except YAMLError:
        return {}, raw_text

    if not isinstance(parsed, dict):
        return {}, raw_text

    normalized = {str(key): value for key, value in parsed.items()}
    return normalized, body.lstrip("\n")


def parse_client_modified_map(payload: str | None) -> dict[str, datetime]:
    """Parse the client-side file last modified map sent by the upload form."""
    if not payload:
        return {}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    modified_map: dict[str, datetime] = {}
    for file_name, value in parsed.items():
        if not isinstance(file_name, str):
            continue
        if not isinstance(value, str):
            continue
        try:
            modified_map[file_name] = datetime.fromisoformat(value)
        except ValueError:
            continue

    return modified_map


def sanitize_embedded_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Drop empty values and normalize datetimes to ISO strings."""
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, datetime):
            sanitized[key] = value.astimezone(UTC).isoformat()
            continue
        sanitized[str(key)] = value
    return sanitized
