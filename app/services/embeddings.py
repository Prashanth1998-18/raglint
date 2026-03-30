"""OpenAI embedding service for chunk analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.errors import MissingAPIKeyError, classify_openai_error
from app.models.document import Chunk


@dataclass(slots=True)
class EmbeddingBatchFailure:
    """A single embedding batch that could not be completed after a retry."""

    batch_number: int
    chunk_count: int
    error_message: str


@dataclass(slots=True)
class EmbeddingRunResult:
    """The outcome of an embedding run, including any partial failures."""

    chunks: list[Chunk]
    embedded_chunk_count: int
    failed_batches: list[EmbeddingBatchFailure] = field(default_factory=list)


class OpenAIEmbeddingService:
    """Attach OpenAI embeddings to chunks in manageable request batches."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.batch_size = batch_size

    async def embed_chunks(self, chunks: list[Chunk], api_key: str) -> list[Chunk]:
        """Return chunk embeddings and preserve already-completed batches on failure."""
        cleaned_api_key = api_key.strip()
        if not cleaned_api_key:
            raise MissingAPIKeyError(
                "An OpenAI API key is required for analysis. Click the settings icon to add your key."
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment setup
            raise RuntimeError("The openai package is required for duplication analysis.") from exc

        missing_embedding_chunks = [chunk for chunk in chunks if chunk.embedding is None]
        if not missing_embedding_chunks:
            return EmbeddingRunResult(chunks=chunks, embedded_chunk_count=len(chunks))

        try:
            client = AsyncOpenAI(api_key=cleaned_api_key, max_retries=0, timeout=60.0)
        except TypeError:
            client = AsyncOpenAI(api_key=cleaned_api_key)
        failed_batches: list[EmbeddingBatchFailure] = []

        for batch_number, batch_start in enumerate(range(0, len(missing_embedding_chunks), self.batch_size), start=1):
            batch = missing_embedding_chunks[batch_start : batch_start + self.batch_size]
            response = None
            last_error: Exception | None = None

            for _attempt in range(2):
                try:
                    response = await client.embeddings.create(
                        model=self.model,
                        input=[chunk.text for chunk in batch],
                    )
                    break
                except Exception as exc:  # pragma: no cover - depends on remote API behavior
                    last_error = exc

            if response is None:
                if last_error is None:
                    continue
                classified_error = classify_openai_error(last_error)
                if not any(chunk.embedding is not None for chunk in chunks):
                    raise classified_error from last_error
                failed_batches.append(
                    EmbeddingBatchFailure(
                        batch_number=batch_number,
                        chunk_count=len(batch),
                        error_message=classified_error.message,
                    )
                )
                continue

            for chunk, item in zip(batch, response.data):
                chunk.embedding = list(item.embedding)

        return EmbeddingRunResult(
            chunks=chunks,
            embedded_chunk_count=sum(1 for chunk in chunks if chunk.embedding is not None),
            failed_batches=failed_batches,
        )
