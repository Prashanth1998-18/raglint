"""Cosine similarity utilities for duplicate and contradiction candidate detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.models.document import Chunk


@dataclass(slots=True)
class SimilarityPair:
    """A pair of chunks with similarity above the configured threshold."""

    left_chunk: Chunk
    right_chunk: Chunk
    similarity_score: float
    comparison_scope: Literal["new_vs_new", "new_vs_existing"]
    severity: Literal["exact", "near_duplicate", "candidate"]


class SimilarityService:
    """Compute pairwise cosine similarity for new and existing chunks."""

    def __init__(
        self,
        *,
        exact_threshold: float = 0.98,
        near_duplicate_threshold: float = 0.85,
    ) -> None:
        if not 0 < near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be between 0 and 1")
        if not 0 < exact_threshold <= 1:
            raise ValueError("exact_threshold must be between 0 and 1")
        if exact_threshold < near_duplicate_threshold:
            raise ValueError("exact_threshold must be greater than or equal to near_duplicate_threshold")

        self.exact_threshold = exact_threshold
        self.near_duplicate_threshold = near_duplicate_threshold

    def find_similar_chunks(
        self,
        new_chunks: list[Chunk],
        existing_chunks: list[Chunk] | None = None,
    ) -> list[SimilarityPair]:
        """Return chunk pairs at or above the near-duplicate threshold."""
        if not new_chunks:
            return []

        self._validate_embeddings(new_chunks, "new chunks")
        existing_chunks = existing_chunks or []
        if existing_chunks:
            self._validate_embeddings(existing_chunks, "existing chunks")

        pairs: list[SimilarityPair] = []

        new_matrix = self._normalized_matrix(new_chunks)
        within_new = new_matrix @ new_matrix.T

        for left_index in range(len(new_chunks)):
            for right_index in range(left_index + 1, len(new_chunks)):
                score = float(within_new[left_index, right_index])
                if score < self.near_duplicate_threshold:
                    continue
                pairs.append(
                    SimilarityPair(
                        left_chunk=new_chunks[left_index],
                        right_chunk=new_chunks[right_index],
                        similarity_score=score,
                        comparison_scope="new_vs_new",
                        severity=self._severity_for_score(score),
                    )
                )

        if existing_chunks:
            existing_matrix = self._normalized_matrix(existing_chunks)
            cross_scores = new_matrix @ existing_matrix.T

            for left_index in range(len(new_chunks)):
                for right_index in range(len(existing_chunks)):
                    score = float(cross_scores[left_index, right_index])
                    if score < self.near_duplicate_threshold:
                        continue
                    pairs.append(
                        SimilarityPair(
                            left_chunk=new_chunks[left_index],
                            right_chunk=existing_chunks[right_index],
                            similarity_score=score,
                            comparison_scope="new_vs_existing",
                            severity=self._severity_for_score(score),
                        )
                    )

        return sorted(pairs, key=lambda pair: pair.similarity_score, reverse=True)

    def find_pairs_in_range(
        self,
        new_chunks: list[Chunk],
        existing_chunks: list[Chunk] | None = None,
        *,
        min_similarity: float,
        max_similarity: float,
    ) -> list[SimilarityPair]:
        """Return chunk pairs whose cosine similarity falls within an inclusive range."""
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        if not 0 <= max_similarity <= 1:
            raise ValueError("max_similarity must be between 0 and 1")
        if min_similarity > max_similarity:
            raise ValueError("min_similarity must be less than or equal to max_similarity")
        if not new_chunks:
            return []

        self._validate_embeddings(new_chunks, "new chunks")
        existing_chunks = existing_chunks or []
        if existing_chunks:
            self._validate_embeddings(existing_chunks, "existing chunks")

        pairs: list[SimilarityPair] = []
        new_matrix = self._normalized_matrix(new_chunks)
        within_new = new_matrix @ new_matrix.T

        for left_index in range(len(new_chunks)):
            for right_index in range(left_index + 1, len(new_chunks)):
                score = float(within_new[left_index, right_index])
                if score < min_similarity or score > max_similarity:
                    continue
                pairs.append(
                    SimilarityPair(
                        left_chunk=new_chunks[left_index],
                        right_chunk=new_chunks[right_index],
                        similarity_score=score,
                        comparison_scope="new_vs_new",
                        severity="candidate",
                    )
                )

        if existing_chunks:
            existing_matrix = self._normalized_matrix(existing_chunks)
            cross_scores = new_matrix @ existing_matrix.T

            for left_index in range(len(new_chunks)):
                for right_index in range(len(existing_chunks)):
                    score = float(cross_scores[left_index, right_index])
                    if score < min_similarity or score > max_similarity:
                        continue
                    pairs.append(
                        SimilarityPair(
                            left_chunk=new_chunks[left_index],
                            right_chunk=existing_chunks[right_index],
                            similarity_score=score,
                            comparison_scope="new_vs_existing",
                            severity="candidate",
                        )
                    )

        return sorted(pairs, key=lambda pair: pair.similarity_score, reverse=True)

    def _severity_for_score(self, score: float) -> Literal["exact", "near_duplicate"]:
        """Map a similarity score to the configured severity level."""
        return "exact" if score >= self.exact_threshold else "near_duplicate"

    def _normalized_matrix(self, chunks: list[Chunk]) -> np.ndarray:
        """Convert chunk embeddings into a normalized matrix."""
        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("All chunk embeddings must share the same vector length.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("All chunk embeddings must be non-zero vectors.")
        return matrix / norms

    def _validate_embeddings(self, chunks: list[Chunk], label: str) -> None:
        """Ensure every chunk has an embedding before similarity is computed."""
        missing = [chunk.id for chunk in chunks if chunk.embedding is None]
        if missing:
            raise ValueError(f"Similarity cannot run because {label} are missing embeddings.")
        dimensions = {len(chunk.embedding or []) for chunk in chunks}
        if len(dimensions) > 1:
            raise ValueError(f"Similarity cannot run because {label} use inconsistent embedding dimensions.")
