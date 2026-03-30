"""Contradiction detection pass for Phase 4."""

from __future__ import annotations

import json
import logging
from math import ceil
from typing import Any

from app.errors import MissingAPIKeyError, classify_openai_error
from app.models.document import (
    ContradictionFinding,
    ContradictionRunStats,
    FindingChunk,
)
from app.services.similarity import SimilarityPair


logger = logging.getLogger(__name__)


class ContradictionDetectionPass:
    """Use an OpenAI chat model to detect contradictions between similar chunks."""

    INPUT_COST_PER_MILLION = 0.30
    OUTPUT_COST_PER_MILLION = 1.20
    DEFAULT_MODEL = "gpt-4o-mini"
    SYSTEM_PROMPT = (
        "You are a document quality analyst. You will be given two text passages. "
        "Your task is to determine if they contain a direct factual contradiction.\n\n"
        "Follow these steps strictly:\n\n"
        "Step 1: Identify the specific factual claims in Passage A. List them.\n"
        "Step 2: Identify the specific factual claims in Passage B. List them.\n"
        "Step 3: For each claim in Passage A, check if Passage B makes a claim about the EXACT same thing "
        "(same policy, same metric, same rule, same entity) but with a different value or conclusion.\n"
        "Step 4: If you found a pair of claims about the exact same thing with incompatible values, "
        "that is a CONTRADICTION. If the claims are about different things, even within the same broad "
        "topic, that is NOT a contradiction.\n\n"
        "Examples of contradictions:\n"
        "- Passage A says hotel limit is $75, Passage B says hotel limit is $100 (same policy, different values)\n"
        "- Passage A says dental starts after 90 days, Passage B says dental starts on day one (same benefit, different timing)\n\n"
        "Examples of NOT contradictions:\n"
        "- Passage A discusses hotel reimbursement, Passage B discusses meal allowances (different policies)\n"
        "- Passage A covers domestic travel rules, Passage B covers international travel rules (different scopes)\n"
        "- Passage A mentions flights, Passage B mentions ground transportation (different topics)\n\n"
        "Be strict. When in doubt, mark as CONSISTENT.\n\n"
        "Respond with JSON only:\n"
        'If contradiction found: {"status": "CONTRADICTION", "claim_a": "the specific claim from A", '
        '"claim_b": "the conflicting claim from B", "explanation": "why these cannot both be true"}\n'
        'If no contradiction: {"status": "CONSISTENT"}'
    )

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_completion_tokens: int = 400,
        include_same_document_pairs: bool = True,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.include_same_document_pairs = include_same_document_pairs

    async def run(
        self,
        candidate_pairs: list[SimilarityPair],
        *,
        api_key: str,
    ) -> tuple[list[ContradictionFinding], ContradictionRunStats]:
        """Analyze contradiction candidate pairs and return findings plus usage stats."""
        if not api_key.strip():
            raise MissingAPIKeyError(
                "An OpenAI API key is required for analysis. Click the settings icon to add your key."
            )

        filtered_pairs = self._filter_candidate_pairs(candidate_pairs)
        stats = ContradictionRunStats(candidate_pairs_considered=len(filtered_pairs))
        if not filtered_pairs:
            return [], stats

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("The openai package is required for contradiction detection.") from exc

        try:
            client = AsyncOpenAI(api_key=api_key.strip(), max_retries=0, timeout=60.0)
        except TypeError:
            client = AsyncOpenAI(api_key=api_key.strip())
        findings: list[ContradictionFinding] = []

        for pair in filtered_pairs:
            user_prompt = self._build_user_prompt(pair)
            estimated_prompt_tokens = self._estimate_tokens(f"{self.SYSTEM_PROMPT}\n{user_prompt}")
            estimated_completion_tokens = self._estimate_tokens('{"status":"CONSISTENT"}')
            stats.llm_calls_made += 1

            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:  # pragma: no cover - API failure path
                stats.failed_calls += 1
                stats.prompt_tokens += estimated_prompt_tokens
                stats.completion_tokens += estimated_completion_tokens
                classified_error = classify_openai_error(exc)
                logger.warning(
                    "Contradiction call failed for %s vs %s: %s",
                    pair.left_chunk.id,
                    pair.right_chunk.id,
                    classified_error.message,
                )
                continue

            usage = getattr(response, "usage", None)
            stats.prompt_tokens += getattr(usage, "prompt_tokens", estimated_prompt_tokens)
            stats.completion_tokens += getattr(usage, "completion_tokens", estimated_completion_tokens)

            parsed = self.parse_llm_response(self._extract_content(response))
            if parsed.get("status") != "CONTRADICTION":
                continue

            finding = self._build_finding(pair, parsed)
            if finding is not None:
                findings.append(finding)

        stats.estimated_cost_usd = round(
            ((stats.prompt_tokens * self.INPUT_COST_PER_MILLION) + (stats.completion_tokens * self.OUTPUT_COST_PER_MILLION))
            / 1_000_000,
            6,
        )

        findings.sort(key=lambda finding: (0 if finding.severity == "high" else 1, -finding.similarity_score))
        return findings, stats

    def parse_llm_response(self, raw_content: str) -> dict[str, Any]:
        """Parse the LLM response into a normalized JSON payload."""
        cleaned = raw_content.strip()
        if not cleaned:
            return {"status": "CONSISTENT"}

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"status": "CONSISTENT"}

        if not isinstance(parsed, dict):
            return {"status": "CONSISTENT"}

        status = str(parsed.get("status", "CONSISTENT")).upper()
        if status != "CONTRADICTION":
            return {"status": "CONSISTENT"}

        claim_a = str(parsed.get("claim_a", "")).strip()
        claim_b = str(parsed.get("claim_b", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        severity = str(parsed.get("severity", "")).strip().upper()
        normalized_severity = self._normalize_severity(severity, claim_a, claim_b, explanation)

        return {
            "status": "CONTRADICTION",
            "claim_a": claim_a,
            "claim_b": claim_b,
            "explanation": explanation or "The passages appear to make incompatible claims.",
            "severity": normalized_severity,
        }

    def _build_user_prompt(self, pair: SimilarityPair) -> str:
        """Create the user prompt for a contradiction comparison call."""
        left_source = pair.left_chunk.parent_document_name or "Unknown source"
        right_source = pair.right_chunk.parent_document_name or "Unknown source"
        return (
            f"Passage A from {left_source}:\n{pair.left_chunk.text}\n\n"
            f"Passage B from {right_source}:\n{pair.right_chunk.text}"
        )

    def _build_finding(
        self,
        pair: SimilarityPair,
        parsed: dict[str, Any],
    ) -> ContradictionFinding | None:
        """Convert a parsed contradiction response into a report finding."""
        claim_a = str(parsed.get("claim_a", "")).strip()
        claim_b = str(parsed.get("claim_b", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        if not claim_a and not claim_b and not explanation:
            return None

        severity = str(parsed.get("severity", "medium")).lower()
        why_it_matters = (
            "Conflicting chunks can cause retrieval to surface incompatible evidence, which forces the model "
            "to guess which claim is authoritative."
        )
        recommendation = (
            "Resolve this contradiction before indexing both passages."
            if severity == "high"
            else "Review these passages and clarify scope, timing, or ownership before indexing both."
        )

        return ContradictionFinding(
            chunks_involved=[
                self._snapshot(pair.left_chunk, source_scope="new"),
                self._snapshot(
                    pair.right_chunk,
                    source_scope="existing" if pair.comparison_scope == "new_vs_existing" else "new",
                ),
            ],
            similarity_score=round(pair.similarity_score, 3),
            severity="high" if severity == "high" else "medium",
            claim_a=claim_a or "Claim not extracted from passage A.",
            claim_b=claim_b or "Claim not extracted from passage B.",
            explanation=explanation or "The passages appear to make conflicting statements.",
            why_it_matters=why_it_matters,
            recommendation=recommendation,
        )

    def _snapshot(self, chunk, *, source_scope: str) -> FindingChunk:
        """Build a report chunk snapshot."""
        return FindingChunk(
            chunk_id=chunk.id,
            text=chunk.text,
            source_filename=chunk.parent_document_name or "Unknown source",
            parent_document_id=chunk.parent_document_id,
            source_scope=source_scope,
        )

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

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count from character length."""
        return max(1, ceil(len(text) / 4))

    def _normalize_severity(
        self,
        severity: str,
        claim_a: str,
        claim_b: str,
        explanation: str,
    ) -> str:
        """Normalize or infer contradiction severity."""
        if severity in {"HIGH", "MEDIUM"}:
            return severity.lower()

        lower_explanation = explanation.lower()
        if any(keyword in lower_explanation for keyword in ("ambiguous", "partial", "may", "might", "unclear")):
            return "medium"
        if claim_a and claim_b:
            return "high"
        return "medium"

    def _filter_candidate_pairs(self, candidate_pairs: list[SimilarityPair]) -> list[SimilarityPair]:
        """Remove candidate pairs that should not be sent to the contradiction model."""
        if self.include_same_document_pairs:
            return candidate_pairs
        return [pair for pair in candidate_pairs if not self._is_same_document_pair(pair)]

    def _is_same_document_pair(self, pair: SimilarityPair) -> bool:
        """Return whether both chunks come from the same source document."""
        left_document_id = (pair.left_chunk.parent_document_id or "").strip()
        right_document_id = (pair.right_chunk.parent_document_id or "").strip()
        if left_document_id and right_document_id and left_document_id == right_document_id:
            return True

        left_document_name = (pair.left_chunk.parent_document_name or "").strip().lower()
        right_document_name = (pair.right_chunk.parent_document_name or "").strip().lower()
        return bool(left_document_name and right_document_name and left_document_name == right_document_name)
