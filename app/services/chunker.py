"""Chunking logic for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.document import Chunk, ChunkPosition, Document


@dataclass(slots=True)
class TextSpan:
    """A contiguous text span with offsets in the original document."""

    start: int
    end: int
    text: str


class RecursiveCharacterChunker:
    """Chunk text by recursively preferring natural separators."""

    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a parsed document into normalized chunks."""
        text = document.text
        if not text.strip():
            return []

        spans = self._split_to_spans(text, 0, self.separators)
        boundaries = sorted({span.end for span in spans if span.end > 0})

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = self._select_chunk_end(start, text_length, boundaries)
            if end <= start:
                break

            normalized_start = start
            while normalized_start < end and text[normalized_start].isspace():
                normalized_start += 1

            normalized_end = end
            while normalized_end > normalized_start and text[normalized_end - 1].isspace():
                normalized_end -= 1

            if normalized_start < normalized_end:
                chunks.append(
                    Chunk(
                        text=text[normalized_start:normalized_end],
                        parent_document_id=document.id,
                        parent_document_name=document.metadata.filename,
                        position=ChunkPosition(
                            chunk_index=chunk_index,
                            start_char=normalized_start,
                            end_char=normalized_end,
                        ),
                        metadata={
                            "chunk_method": "recursive_character",
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                )
                chunk_index += 1

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _select_chunk_end(self, start: int, text_length: int, boundaries: list[int]) -> int:
        """Pick the furthest preferred boundary that respects chunk size."""
        max_end = min(start + self.chunk_size, text_length)
        candidate_end = max_end

        for boundary in boundaries:
            if boundary <= start:
                continue
            if boundary > max_end:
                break
            candidate_end = boundary

        return candidate_end

    def _split_to_spans(
        self,
        text: str,
        start_offset: int,
        separators: list[str],
    ) -> list[TextSpan]:
        """Recursively split text into spans no larger than the chunk size."""
        if len(text) <= self.chunk_size:
            return [TextSpan(start=start_offset, end=start_offset + len(text), text=text)]

        if not separators:
            return self._split_fixed_width(text, start_offset)

        separator = separators[0]
        pieces = self._split_with_separator(text, separator)

        if len(pieces) == 1:
            return self._split_to_spans(text, start_offset, separators[1:])

        spans: list[TextSpan] = []
        cursor = start_offset

        for piece in pieces:
            if not piece:
                continue
            piece_start = cursor
            piece_end = cursor + len(piece)
            cursor = piece_end

            if len(piece) <= self.chunk_size:
                spans.append(TextSpan(start=piece_start, end=piece_end, text=piece))
                continue

            spans.extend(self._split_to_spans(piece, piece_start, separators[1:]))

        return spans

    def _split_fixed_width(self, text: str, start_offset: int) -> list[TextSpan]:
        """Fallback split when no natural separator remains."""
        spans: list[TextSpan] = []
        for offset in range(0, len(text), self.chunk_size):
            end = min(offset + self.chunk_size, len(text))
            spans.append(
                TextSpan(
                    start=start_offset + offset,
                    end=start_offset + end,
                    text=text[offset:end],
                )
            )
        return spans

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        """Split while retaining the separator on the preceding segment."""
        if separator == "":
            return list(text)

        pieces: list[str] = []
        search_start = 0

        while True:
            index = text.find(separator, search_start)
            if index == -1:
                break
            end = index + len(separator)
            pieces.append(text[search_start:end])
            search_start = end

        if search_start < len(text):
            pieces.append(text[search_start:])

        return pieces
