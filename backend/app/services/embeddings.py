from __future__ import annotations

import asyncio
import hashlib
import math
import struct
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import UUID, uuid4

from app.core.config import settings
from app.models.evidence import EvidenceCreate
from app.models.section import Section
from app.services.evidence import create_evidence, delete_evidence_for_paper
from app.services.sections import list_sections
from app.utils.text_sanitize import sanitize_text

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient  # type: ignore[import]
    from qdrant_client.http import models as qmodels  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]

SNIPPET_MAX_CHARS = 400


class VectorStoreNotAvailableError(RuntimeError):
    """Raised when the Qdrant vector store cannot be reached."""


@dataclass
class PreparedSection:
    section_id: UUID
    paper_id: UUID
    title: Optional[str]
    content: str
    snippet: str
    page_number: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    text_hash: str

    def payload(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "paper_id": str(self.paper_id),
            "section_id": str(self.section_id),
            "section_title": self.title,
            "content": self.content,
            "snippet": self.snippet,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text_hash": self.text_hash,
        }
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class VectorRecord:
    vector_id: str
    vector: List[float]
    payload: Dict[str, Any]


@dataclass
class VectorSearchResult:
    vector_id: str
    score: float
    payload: Dict[str, Any]


def build_embedding_backend(model_name: str, dimension: int) -> EmbeddingBackend:
    if SentenceTransformer is not None:
        try:
            return SentenceTransformerBackend(model_name)
        except Exception as exc:  # pragma: no cover - fallback logging
            print(f"[EmbeddingService] Failed to initialize SentenceTransformer: {exc}")
    print("[EmbeddingService] Using deterministic hash embedding backend")
    return HashEmbeddingBackend(dimension)


class EmbeddingBackend:
    async def embed(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:  # pragma: no cover - safety guard
            raise RuntimeError("sentence-transformers is not installed")
        self._model_name = model_name
        self._model: SentenceTransformer | None = None  # type: ignore[valid-type]
        self._lock = threading.Lock()

    def _load_model(self) -> SentenceTransformer:  # type: ignore[override]
        with self._lock:
            if self._model is None:
                self._model = SentenceTransformer(self._model_name)  # type: ignore[call-arg,assignment]
            return self._model

    def _encode(self, texts: List[str], batch_size: int) -> List[List[float]]:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [list(map(float, vector)) for vector in embeddings]

    async def embed(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        data = list(texts)
        if not data:
            return []
        return await asyncio.to_thread(self._encode, data, batch_size)


class HashEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dimension: int) -> None:
        self._dimension = max(1, dimension)

    async def embed(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:  # noqa: ARG002 - batch size unused
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> List[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        required_bytes = self._dimension * 4
        buffer = bytearray()
        chunk = seed
        while len(buffer) < required_bytes:
            buffer.extend(chunk)
            chunk = hashlib.sha256(chunk).digest()
        floats = struct.unpack(f"<{self._dimension}f", buffer[:required_bytes])
        norm = math.sqrt(sum(value * value for value in floats))
        if norm == 0:
            return [0.0 for _ in floats]
        return [float(value / norm) for value in floats]


class EmbeddingCacheRepository:
    async def get_many(self, model: str, hashes: Sequence[str]) -> Dict[str, List[float]]:
        if not hashes:
            return {}
        pool = get_pool()
        query = """
            SELECT text_hash, embedding
            FROM embedding_cache
            WHERE model = $1 AND text_hash = ANY($2::text[])
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, model, list(hashes))
        return {row["text_hash"]: list(row["embedding"]) for row in rows}

    async def upsert_many(self, model: str, values: Mapping[str, Sequence[float]]) -> None:
        if not values:
            return
        pool = get_pool()
        records = [
            (model, text_hash, list(map(float, embedding)))
            for text_hash, embedding in values.items()
        ]
        query = """
            INSERT INTO embedding_cache (model, text_hash, embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (model, text_hash)
            DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW()
        """
        async with pool.acquire() as conn:
            await conn.executemany(query, records)


class VectorStore:
    def __init__(self, collection_name: str, dimension: int) -> None:
        self._collection_name = collection_name
        self._dimension = dimension
        self._client: QdrantClient | None = None  # type: ignore[valid-type]
        self._collection_ready = False
        self._lock = threading.Lock()

    async def replace_for_paper(self, paper_id: UUID, records: Sequence[VectorRecord], batch_size: int) -> None:
        client = await self._ensure_client()
        await self._ensure_collection(client)
        await asyncio.to_thread(
            client.delete,
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(  # type: ignore[union-attr]
                filter=qmodels.Filter(  # type: ignore[union-attr]
                    must=[
                        qmodels.FieldCondition(  # type: ignore[union-attr]
                            key="paper_id",
                            match=qmodels.MatchValue(value=str(paper_id)),  # type: ignore[union-attr]
                        )
                    ]
                )
            ),
        )
        if not records:
            return
        for batch in _chunk_sequence(records, max(1, batch_size)):
            points = [
                qmodels.PointStruct(  # type: ignore[union-attr]
                    id=record.vector_id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in batch
            ]
            await asyncio.to_thread(
                client.upsert,
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

    async def search(self, vector: Sequence[float], limit: int) -> List[VectorSearchResult]:
        client = await self._ensure_client()
        await self._ensure_collection(client)
        if limit <= 0:
            return []
        try:
            results = await asyncio.to_thread(
                client.search,
                collection_name=self._collection_name,
                query_vector=list(map(float, vector)),
                limit=max(1, limit),
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreNotAvailableError(str(exc)) from exc

        normalized: List[VectorSearchResult] = []
        for item in results:
            payload = dict(getattr(item, "payload", {}) or {})
            normalized.append(
                VectorSearchResult(
                    vector_id=str(getattr(item, "id", "")),
                    score=float(getattr(item, "score", 0.0) or 0.0),
                    payload=payload,
                )
            )
        return normalized

    async def delete_for_paper(self, paper_id: UUID) -> None:
        client = await self._ensure_client()
        await self._ensure_collection(client)
        await asyncio.to_thread(
            client.delete,
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(  # type: ignore[union-attr]
                filter=qmodels.Filter(  # type: ignore[union-attr]
                    must=[
                        qmodels.FieldCondition(  # type: ignore[union-attr]
                            key="paper_id",
                            match=qmodels.MatchValue(value=str(paper_id)),  # type: ignore[union-attr]
                        )
                    ]
                )
            ),
        )

    async def _ensure_client(self) -> QdrantClient:
        if QdrantClient is None or qmodels is None:
            raise VectorStoreNotAvailableError("qdrant-client is not installed")

        def _create_client() -> QdrantClient:
            with self._lock:
                if self._client is None:
                    self._client = QdrantClient(  # type: ignore[call-arg,assignment]
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=10.0,
                    )
                return self._client

        return await asyncio.to_thread(_create_client)

    async def _ensure_collection(self, client: QdrantClient) -> None:
        if qmodels is None:
            raise VectorStoreNotAvailableError("qdrant-client is not installed")
        if self._collection_ready:
            return

        def _prepare() -> None:
            try:
                client.get_collection(self._collection_name)
            except Exception:
                try:
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=qmodels.VectorParams(  # type: ignore[union-attr]
                            size=self._dimension,
                            distance=qmodels.Distance.COSINE,  # type: ignore[union-attr]
                        ),
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    # If the collection already exists due to a race, ignore the error
                    if "exists" not in str(exc).lower():
                        raise

        await asyncio.to_thread(_prepare)
        self._collection_ready = True


_service: EmbeddingService | None = None
_service_lock = threading.Lock()


class EmbeddingService:
    def __init__(
        self,
        *,
        backend: EmbeddingBackend | None = None,
        cache: EmbeddingCacheRepository | None = None,
        vector_store: VectorStore | None = None,
        sections_fetcher: Callable[..., Awaitable[List[Section]]] | None = None,
        create_evidence_fn: Callable[[EvidenceCreate], Awaitable[Any]] | None = None,
        delete_evidence_fn: Callable[[UUID], Awaitable[None]] | None = None,
    ) -> None:
        self._model_name = settings.embedding_model_name
        self._dimension = max(1, settings.embedding_dimension)
        self._batch_size = max(1, settings.embedding_batch_size)
        self._upsert_batch_size = max(1, settings.qdrant_upsert_batch_size)
        self._backend = backend or build_embedding_backend(
            self._model_name, self._dimension
        )
        self._cache = cache or EmbeddingCacheRepository()
        self._vector_store = vector_store or VectorStore(
            settings.qdrant_collection_name, self._dimension
        )
        self._list_sections = sections_fetcher or list_sections
        self._create_evidence = create_evidence_fn or create_evidence
        self._delete_evidence = delete_evidence_fn or delete_evidence_for_paper

    async def embed_paper_sections(self, paper_id: UUID) -> None:
        sections = await self._load_sections(paper_id)
        prepared = self._prepare_sections(sections)
        if not prepared:
            try:
                await self._vector_store.delete_for_paper(paper_id)
            except VectorStoreNotAvailableError as exc:
                print(f"[EmbeddingService] Vector store unavailable: {exc}")
            await self._delete_evidence(paper_id)
            return
        try:
            embeddings = await self._embed_prepared_sections(prepared)
        except Exception as exc:
            print(f"[EmbeddingService] Failed to compute embeddings for paper {paper_id}: {exc}")
            return

        vector_records: List[VectorRecord] = []
        evidence_records: List[EvidenceCreate] = []
        for section in prepared:
            vector = embeddings.get(section.text_hash)
            if vector is None:
                continue
            if len(vector) != self._dimension:
                print(
                    "[EmbeddingService] Skipping section due to dimension mismatch: "
                    f"expected {self._dimension}, got {len(vector)}",
                )
                continue
            vector_id = uuid4().hex
            vector_records.append(
                VectorRecord(
                    vector_id=vector_id,
                    vector=list(map(float, vector)),
                    payload=section.payload(),
                )
            )
            cleaned_snippet = sanitize_text(section.snippet)
            if not cleaned_snippet:
                print(
                    f"[EmbeddingService] Skipping evidence for section {section.section_id}"
                    " due to empty snippet after sanitization"
                )
                continue
            evidence_records.append(
                EvidenceCreate(
                    paper_id=paper_id,
                    section_id=section.section_id,
                    concept_id=None,
                    relation_id=None,
                    snippet=cleaned_snippet,
                    vector_id=vector_id,
                    embedding_model=self._model_name,
                    score=None,
                    metadata={
                        "text_hash": section.text_hash,
                        "page_number": section.page_number,
                        "char_start": section.char_start,
                        "char_end": section.char_end,
                    },
                )
            )

        if not vector_records:
            try:
                await self._vector_store.delete_for_paper(paper_id)
            except VectorStoreNotAvailableError as exc:
                print(f"[EmbeddingService] Vector store unavailable: {exc}")
                return
            await self._delete_evidence(paper_id)
            return

        try:
            await self._vector_store.replace_for_paper(paper_id, vector_records, self._upsert_batch_size)
        except VectorStoreNotAvailableError as exc:
            print(f"[EmbeddingService] Vector store unavailable: {exc}")
            return
        except Exception as exc:
            print(f"[EmbeddingService] Failed to upsert vectors for paper {paper_id}: {exc}")
            return

        await self._delete_evidence(paper_id)
        for record in evidence_records:
            try:
                await self._create_evidence(record)
            except Exception as exc:
                print(f"[EmbeddingService] Failed to persist evidence: {exc}")
                break

    async def _load_sections(self, paper_id: UUID) -> List[Section]:
        sections: List[Section] = []
        offset = 0
        page_size = 200
        while True:
            batch = await self._list_sections(
                paper_id=paper_id, limit=page_size, offset=offset
            )
            if not batch:
                break
            sections.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)
        return sections

    def _prepare_sections(self, sections: Sequence[Section]) -> List[PreparedSection]:
        prepared: List[PreparedSection] = []
        for section in sections:
            content = sanitize_text(section.content)
            if not content:
                continue
            snippet_source = section.snippet or _build_snippet(content)
            snippet = sanitize_text(snippet_source)
            if not snippet:
                continue
            text_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            prepared.append(
                PreparedSection(
                    section_id=section.id,
                    paper_id=section.paper_id,
                    title=section.title,
                    content=content,
                    snippet=snippet,
                    page_number=section.page_number,
                    char_start=section.char_start,
                    char_end=section.char_end,
                    text_hash=text_hash,
                )
            )
        return prepared

    async def _embed_prepared_sections(
        self, sections: Sequence[PreparedSection]
    ) -> Dict[str, List[float]]:
        unique_texts: Dict[str, str] = {}
        for section in sections:
            if section.text_hash not in unique_texts:
                unique_texts[section.text_hash] = section.content

        cached = await self._cache.get_many(self._model_name, list(unique_texts.keys()))
        missing_hashes = [text_hash for text_hash in unique_texts.keys() if text_hash not in cached]

        newly_computed: Dict[str, List[float]] = {}
        if missing_hashes:
            for batch in _chunk_sequence(missing_hashes, self._batch_size):
                texts = [unique_texts[text_hash] for text_hash in batch]
                vectors = await self._backend.embed(texts, self._batch_size)
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding backend returned mismatched vector count")
                for text_hash, vector in zip(batch, vectors):
                    newly_computed[text_hash] = list(map(float, vector))
            await self._cache.upsert_many(self._model_name, newly_computed)

        combined = dict(cached)
        combined.update(newly_computed)
        return combined


def _build_snippet(content: str) -> Optional[str]:
    cleaned = " ".join(content.split())
    if not cleaned:
        return None
    if len(cleaned) <= SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[:SNIPPET_MAX_CHARS].rstrip() + "…"


def _chunk_sequence(sequence: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    step = max(1, chunk_size)
    for index in range(0, len(sequence), step):
        yield sequence[index : index + step]


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EmbeddingService()
    return _service


async def embed_paper_sections(paper_id: UUID) -> None:
    service = get_embedding_service()
    await service.embed_paper_sections(paper_id)

from app.db.pool import get_pool
