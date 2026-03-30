"""Duplication detection pass built on top of similarity pairs."""

from __future__ import annotations

from collections import defaultdict

from app.models.document import Chunk, DuplicationFinding, FindingChunk
from app.services.similarity import SimilarityPair


class DuplicationDetectionPass:
    """Build report-ready duplication findings from similarity pairs."""

    def build_findings(self, pairs: list[SimilarityPair]) -> list[DuplicationFinding]:
        """Return exact duplicate clusters, near duplicates, and brownfield matches."""
        if not pairs:
            return []

        exact_pairs = [pair for pair in pairs if pair.comparison_scope == "new_vs_new" and pair.severity == "exact"]
        near_pairs = [pair for pair in pairs if pair.comparison_scope == "new_vs_new" and pair.severity == "near_duplicate"]
        brownfield_pairs = [pair for pair in pairs if pair.comparison_scope == "new_vs_existing"]

        findings: list[DuplicationFinding] = []
        exact_findings, clustered_chunk_ids = self._build_exact_cluster_findings(exact_pairs)
        findings.extend(exact_findings)
        near_pairs = [
            pair
            for pair in near_pairs
            if pair.left_chunk.id not in clustered_chunk_ids or pair.right_chunk.id not in clustered_chunk_ids
        ]
        findings.extend(self._build_brownfield_findings(brownfield_pairs))
        findings.extend(self._build_near_duplicate_findings(near_pairs))
        return findings

    def _build_exact_cluster_findings(
        self,
        pairs: list[SimilarityPair],
    ) -> tuple[list[DuplicationFinding], set[str]]:
        """Collapse exact duplicate pairs into connected clusters."""
        if not pairs:
            return [], set()

        parent: dict[str, str] = {}
        chunk_lookup: dict[str, Chunk] = {}
        pair_scores_by_root: dict[str, list[float]] = defaultdict(list)

        def find(chunk_id: str) -> str:
            root = parent.setdefault(chunk_id, chunk_id)
            if root != chunk_id:
                parent[chunk_id] = find(root)
            return parent[chunk_id]

        def union(left_id: str, right_id: str) -> None:
            left_root = find(left_id)
            right_root = find(right_id)
            if left_root != right_root:
                parent[right_root] = left_root

        for pair in pairs:
            chunk_lookup[pair.left_chunk.id] = pair.left_chunk
            chunk_lookup[pair.right_chunk.id] = pair.right_chunk
            union(pair.left_chunk.id, pair.right_chunk.id)

        clusters: dict[str, list[Chunk]] = defaultdict(list)
        for chunk_id, chunk in chunk_lookup.items():
            clusters[find(chunk_id)].append(chunk)

        for pair in pairs:
            pair_scores_by_root[find(pair.left_chunk.id)].append(pair.similarity_score)

        findings: list[DuplicationFinding] = []
        clustered_chunk_ids: set[str] = set()
        for root, chunks in clusters.items():
            if len(chunks) < 2:
                continue
            clustered_chunk_ids.update(chunk.id for chunk in chunks)
            findings.append(
                DuplicationFinding(
                    finding_type="exact",
                    chunks_involved=[
                        self._finding_chunk(chunk, source_scope="new") for chunk in self._sort_chunks(chunks)
                    ],
                    similarity_score=min(pair_scores_by_root[root]),
                    explanation=(
                        f"{len(chunks)} chunks are effectively the same content. Keeping all of them wastes retrieval "
                        "space and can crowd better context out of the prompt."
                    ),
                    recommendation="Keep the strongest source of truth and remove or suppress the redundant copies.",
                )
            )

        return sorted(findings, key=lambda finding: finding.similarity_score, reverse=True), clustered_chunk_ids

    def _build_near_duplicate_findings(self, pairs: list[SimilarityPair]) -> list[DuplicationFinding]:
        """Create review findings for near duplicate chunk pairs."""
        findings: list[DuplicationFinding] = []
        for pair in pairs:
            findings.append(
                DuplicationFinding(
                    finding_type="near_duplicate",
                    chunks_involved=[
                        self._finding_chunk(pair.left_chunk, source_scope="new"),
                        self._finding_chunk(pair.right_chunk, source_scope="new"),
                    ],
                    similarity_score=pair.similarity_score,
                    explanation=(
                        "These chunks cover very similar information with slightly different wording. "
                        "That can lead to redundant retrieval results and inconsistent answers."
                    ),
                    recommendation="Review both sources and decide whether to merge, remove, or keep both with clearer scope.",
                )
            )

        return findings

    def _build_brownfield_findings(self, pairs: list[SimilarityPair]) -> list[DuplicationFinding]:
        """Create already-in-index findings for new versus existing chunk matches."""
        findings: list[DuplicationFinding] = []
        for pair in pairs:
            findings.append(
                DuplicationFinding(
                    finding_type="already_in_index",
                    chunks_involved=[
                        self._finding_chunk(pair.left_chunk, source_scope="new"),
                        self._finding_chunk(pair.right_chunk, source_scope="existing"),
                    ],
                    similarity_score=pair.similarity_score,
                    explanation=(
                        "The new chunk matches content already present in the existing index. "
                        "Adding it again can increase duplication and retrieval noise."
                    ),
                    recommendation="Review whether this new content adds value or if the existing indexed chunk is already sufficient.",
                )
            )

        return findings

    def _finding_chunk(self, chunk: Chunk, *, source_scope: str) -> FindingChunk:
        """Convert a chunk model into the report-friendly finding shape."""
        return FindingChunk(
            chunk_id=chunk.id,
            text=chunk.text,
            source_filename=chunk.parent_document_name or "Unknown source",
            parent_document_id=chunk.parent_document_id,
            source_scope=source_scope,
        )

    def _sort_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Keep cluster display stable across runs."""
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.parent_document_name or "",
                chunk.position.chunk_index,
                chunk.id,
            ),
        )
