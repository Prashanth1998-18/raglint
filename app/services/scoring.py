"""Corpus health scoring for the final Phase 5 report."""

from __future__ import annotations

from app.models.document import (
    Chunk,
    ContradictionFinding,
    ContradictionRunStats,
    CorpusHealthScore,
    CorpusHealthSummary,
    Document,
    DuplicationFinding,
    HealthDimensionScore,
    MetadataAuditSummary,
    MetadataFinding,
    ROTFinding,
    StalenessFinding,
)


class CorpusHealthScorer:
    """Calculate the overall and per-dimension health of the current report."""

    DEFAULT_WEIGHTS = {
        "duplication": 0.25,
        "staleness": 0.20,
        "contradictions": 0.25,
        "metadata": 0.10,
        "rot": 0.20,
    }

    def calculate(
        self,
        *,
        documents: list[Document],
        chunks: list[Chunk],
        duplication_findings: list[DuplicationFinding],
        staleness_findings: list[StalenessFinding],
        contradiction_findings: list[ContradictionFinding],
        contradiction_stats: ContradictionRunStats,
        metadata_findings: list[MetadataFinding],
        metadata_summary: MetadataAuditSummary,
        rot_findings: list[ROTFinding],
        duplication_score_available: bool = True,
        staleness_score_available: bool = True,
        contradiction_score_available: bool,
        metadata_score_available: bool = True,
        rot_score_available: bool = True,
    ) -> CorpusHealthScore:
        """Compute report health scores and a projected score after removals."""
        duplicated_chunk_ids = self._duplicated_chunk_ids(duplication_findings) if duplication_score_available else set()
        stale_chunk_ids = {
            finding.chunk.chunk_id
            for finding in staleness_findings
            if finding.staleness_score > 0.7
        } if staleness_score_available else set()
        removable_document_ids = {
            finding.document_id
            for finding in rot_findings
            if finding.recommendation.lower().startswith("remove")
        } if rot_score_available else set()
        documents_with_issues = self._documents_with_issues(
            documents=documents,
            duplication_findings=duplication_findings,
            staleness_findings=staleness_findings,
            contradiction_findings=contradiction_findings,
            metadata_findings=metadata_findings,
            rot_findings=rot_findings,
            duplication_available=duplication_score_available,
            staleness_available=staleness_score_available,
            contradiction_available=contradiction_score_available,
            metadata_available=metadata_score_available,
            rot_available=rot_score_available,
        )

        duplication_score = (
            self._duplication_score(len(chunks), len(duplicated_chunk_ids))
            if duplication_score_available
            else None
        )
        staleness_score = self._staleness_score(staleness_findings) if staleness_score_available else None
        contradiction_score = self._contradiction_score(
            contradiction_findings,
            contradiction_stats,
            contradiction_score_available,
        )
        metadata_score = round(metadata_summary.average_completeness * 100, 1) if metadata_score_available else None
        rot_score = self._rot_score(rot_findings) if rot_score_available else None
        overall_score = self._weighted_average(
            {
                "duplication": duplication_score,
                "staleness": staleness_score,
                "contradictions": contradiction_score,
                "metadata": metadata_score,
                "rot": rot_score,
            }
        )

        remaining_chunks = [
            chunk
            for chunk in chunks
            if chunk.parent_document_id not in removable_document_ids
            and chunk.id not in duplicated_chunk_ids
            and chunk.id not in stale_chunk_ids
        ]
        remaining_staleness_findings = [
            finding
            for finding in staleness_findings
            if finding.chunk.parent_document_id not in removable_document_ids
            and finding.chunk.chunk_id not in stale_chunk_ids
        ]
        remaining_contradictions = [
            finding
            for finding in contradiction_findings
            if not any(
                chunk.parent_document_id in removable_document_ids
                for chunk in finding.chunks_involved
            )
        ]
        remaining_rot_findings = [
            finding
            for finding in rot_findings
            if finding.document_id not in removable_document_ids
        ]
        remaining_metadata_scores: list[float] = []
        for finding in remaining_rot_findings:
            metadata = finding.signals.get("metadata")
            if not isinstance(metadata, dict):
                continue
            score = metadata.get("completeness_score")
            if isinstance(score, (int, float)):
                remaining_metadata_scores.append(float(score))

        projected_contradiction_score: float | None
        if not contradiction_score_available:
            projected_contradiction_score = None
        elif not remaining_contradictions:
            projected_contradiction_score = 100.0
        else:
            projected_contradiction_score = self._contradiction_score(
                remaining_contradictions,
                contradiction_stats,
                contradiction_score_available,
            )

        projected_overall_score = self._weighted_average(
            {
                "duplication": self._duplication_score(len(remaining_chunks), 0) if duplication_score_available else None,
                "staleness": self._staleness_score(remaining_staleness_findings) if staleness_score_available else None,
                "contradictions": projected_contradiction_score,
                "metadata": round(
                    (sum(remaining_metadata_scores) / len(remaining_metadata_scores)) * 100,
                    1,
                ) if metadata_score_available and remaining_metadata_scores else (
                    100.0 if metadata_score_available and documents and not remaining_rot_findings else metadata_score
                ),
                "rot": (
                    self._rot_score(remaining_rot_findings) if remaining_rot_findings else 100.0
                ) if rot_score_available else None,
            }
        )

        summary = CorpusHealthSummary(
            total_chunks_analyzed=len(chunks),
            total_documents=len(documents),
            unique_documents_with_issues=len(documents_with_issues),
            duplicate_clusters=sum(
                1
                for finding in duplication_findings
                if finding.finding_type in {"exact", "near_duplicate"}
            ) if duplication_score_available else 0,
            stale_chunks=len(stale_chunk_ids) if staleness_score_available else 0,
            contradictions_found=len(contradiction_findings) if contradiction_score_available else 0,
            average_metadata_completeness=metadata_summary.average_completeness if metadata_score_available else 0.0,
            documents_recommended_for_removal=len(removable_document_ids) if rot_score_available else 0,
        )

        dimension_scores = [
            HealthDimensionScore(
                key="duplication",
                label="Duplication",
                score=duplication_score,
                weight=self.DEFAULT_WEIGHTS["duplication"],
            ),
            HealthDimensionScore(
                key="staleness",
                label="Staleness",
                score=staleness_score,
                weight=self.DEFAULT_WEIGHTS["staleness"],
            ),
            HealthDimensionScore(
                key="contradictions",
                label="Contradictions",
                score=contradiction_score,
                weight=self.DEFAULT_WEIGHTS["contradictions"],
            ),
            HealthDimensionScore(
                key="metadata",
                label="Metadata",
                score=metadata_score,
                weight=self.DEFAULT_WEIGHTS["metadata"],
            ),
            HealthDimensionScore(
                key="rot",
                label="ROT",
                score=rot_score,
                weight=self.DEFAULT_WEIGHTS["rot"],
            ),
        ]

        return CorpusHealthScore(
            overall_score=overall_score,
            projected_overall_score=projected_overall_score,
            duplication_score=duplication_score,
            staleness_score=staleness_score,
            contradiction_score=contradiction_score,
            metadata_score=metadata_score,
            rot_score=rot_score,
            dimension_scores=dimension_scores,
            skipped_dimensions=[
                label
                for label, available in {
                    "duplication": duplication_score_available,
                    "staleness": staleness_score_available,
                    "contradictions": contradiction_score_available,
                    "metadata": metadata_score_available,
                    "rot": rot_score_available,
                }.items()
                if not available
            ],
            summary=summary,
        )

    def _duplicated_chunk_ids(self, findings: list[DuplicationFinding]) -> set[str]:
        """Collect new chunk identifiers involved in duplication findings."""
        duplicated: set[str] = set()
        for finding in findings:
            for chunk in finding.chunks_involved:
                if chunk.source_scope == "new":
                    duplicated.add(chunk.chunk_id)
        return duplicated

    def _duplication_score(self, total_chunks: int, duplicated_chunks: int) -> float:
        """Score how much of the corpus remains free of duplication."""
        if total_chunks <= 0:
            return 100.0
        score = (1 - (duplicated_chunks / total_chunks)) * 100
        return round(max(0.0, score), 1)

    def _staleness_score(self, findings: list[StalenessFinding]) -> float:
        """Invert average staleness so higher is better."""
        if not findings:
            return 100.0
        average_staleness = sum(finding.staleness_score for finding in findings) / len(findings)
        return round(max(0.0, (1 - average_staleness) * 100), 1)

    def _contradiction_score(
        self,
        findings: list[ContradictionFinding],
        stats: ContradictionRunStats,
        available: bool,
    ) -> float | None:
        """Score contradiction health from findings versus candidate pairs checked."""
        if not available:
            return None
        if stats.candidate_pairs_considered <= 0:
            return 100.0
        score = (1 - (len(findings) / stats.candidate_pairs_considered)) * 100
        return round(max(0.0, score), 1)

    def _rot_score(self, findings: list[ROTFinding]) -> float:
        """Score the percentage of uploaded documents classified as healthy."""
        if not findings:
            return 100.0
        healthy_documents = sum(1 for finding in findings if finding.classifications == ["healthy"])
        return round((healthy_documents / len(findings)) * 100, 1)

    def _weighted_average(self, values: dict[str, float | None]) -> float:
        """Average dimension scores while redistributing missing contradiction weight."""
        active_weights = {
            key: weight
            for key, weight in self.DEFAULT_WEIGHTS.items()
            if values.get(key) is not None
        }
        if not active_weights:
            return 0.0

        total_weight = sum(active_weights.values())
        normalized_weights = {
            key: weight / total_weight
            for key, weight in active_weights.items()
        }
        score = sum((values[key] or 0.0) * normalized_weights[key] for key in active_weights)
        return round(score, 1)

    def _documents_with_issues(
        self,
        *,
        documents: list[Document],
        duplication_findings: list[DuplicationFinding],
        staleness_findings: list[StalenessFinding],
        contradiction_findings: list[ContradictionFinding],
        metadata_findings: list[MetadataFinding],
        rot_findings: list[ROTFinding],
        duplication_available: bool,
        staleness_available: bool,
        contradiction_available: bool,
        metadata_available: bool,
        rot_available: bool,
    ) -> set[str]:
        """Collect uploaded document identifiers that appear in at least one issue list."""
        uploaded_document_ids = {document.id for document in documents}
        issue_document_ids: set[str] = set()

        if duplication_available:
            for finding in duplication_findings:
                for chunk in finding.chunks_involved:
                    if chunk.source_scope == "new" and chunk.parent_document_id in uploaded_document_ids:
                        issue_document_ids.add(chunk.parent_document_id)

        if staleness_available:
            for finding in staleness_findings:
                if finding.chunk.parent_document_id in uploaded_document_ids:
                    issue_document_ids.add(finding.chunk.parent_document_id)

        if contradiction_available:
            for finding in contradiction_findings:
                for chunk in finding.chunks_involved:
                    if chunk.parent_document_id in uploaded_document_ids:
                        issue_document_ids.add(chunk.parent_document_id)

        if metadata_available:
            for finding in metadata_findings:
                if (
                    finding.document_id in uploaded_document_ids
                    and (finding.missing_fields or finding.consistency_issues or finding.completeness_score < 1.0)
                ):
                    issue_document_ids.add(finding.document_id)

        if rot_available:
            for finding in rot_findings:
                if finding.document_id in uploaded_document_ids and finding.classifications != ["healthy"]:
                    issue_document_ids.add(finding.document_id)

        return issue_document_ids
