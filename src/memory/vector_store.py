"""
Vector store for semantic search and retrieval.
Supports multiple embedding backends and storage options.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import json
import numpy as np
from pathlib import Path


@dataclass
class VectorDocument:
    """A document with vector embedding."""
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )


@dataclass
class SearchResult:
    """Result from vector search."""
    document: VectorDocument
    score: float
    rank: int


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = 1536 if "small" in model else 3072

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch."""
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, text: str) -> List[float]:
        """Generate embedding locally."""
        import asyncio
        loop = asyncio.get_event_loop()
        model = self._get_model()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch."""
        import asyncio
        loop = asyncio.get_event_loop()
        model = self._get_model()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class VectorStore:
    """
    Vector store for document storage and similarity search.
    Supports in-memory and persistent storage.
    """

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None,
                 persist_path: Optional[str] = None):
        self.embedding_provider = embedding_provider
        self.persist_path = persist_path
        self.documents: Dict[str, VectorDocument] = {}
        self._index: Optional[Any] = None
        self._index_dirty = True

        if persist_path:
            self._load_from_disk()

    async def add(self, content: str, metadata: Optional[Dict] = None,
                 doc_id: Optional[str] = None) -> VectorDocument:
        """Add a document to the store."""
        doc_id = doc_id or str(uuid.uuid4())

        # Generate embedding
        embedding = None
        if self.embedding_provider:
            embedding = await self.embedding_provider.embed(content)

        doc = VectorDocument(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        self.documents[doc_id] = doc
        self._index_dirty = True

        if self.persist_path:
            self._save_to_disk()

        return doc

    async def add_batch(self, items: List[Dict[str, Any]]) -> List[VectorDocument]:
        """Add multiple documents."""
        docs = []

        if self.embedding_provider:
            # Batch embed
            contents = [item["content"] for item in items]
            embeddings = await self.embedding_provider.embed_batch(contents)

            for item, embedding in zip(items, embeddings):
                doc_id = item.get("doc_id", str(uuid.uuid4()))
                doc = VectorDocument(
                    doc_id=doc_id,
                    content=item["content"],
                    embedding=embedding,
                    metadata=item.get("metadata", {})
                )
                self.documents[doc_id] = doc
                docs.append(doc)
        else:
            for item in items:
                doc = await self.add(
                    content=item["content"],
                    metadata=item.get("metadata"),
                    doc_id=item.get("doc_id")
                )
                docs.append(doc)

        self._index_dirty = True

        if self.persist_path:
            self._save_to_disk()

        return docs

    async def search(self, query: str, k: int = 5,
                    filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents."""
        if not self.documents:
            return []

        # Get query embedding
        if self.embedding_provider:
            query_embedding = await self.embedding_provider.embed(query)
        else:
            # Fall back to simple keyword search
            return self._keyword_search(query, k, filter_metadata)

        # Compute similarities
        results = []
        for doc in self.documents.values():
            if doc.embedding is None:
                continue

            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(doc.metadata, filter_metadata):
                    continue

            score = self._cosine_similarity(query_embedding, doc.embedding)
            results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [
            SearchResult(document=doc, score=score, rank=i+1)
            for i, (doc, score) in enumerate(results[:k])
        ]

    def _keyword_search(self, query: str, k: int,
                       filter_metadata: Optional[Dict]) -> List[SearchResult]:
        """Simple keyword search fallback."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for doc in self.documents.values():
            if filter_metadata and not self._matches_filter(doc.metadata, filter_metadata):
                continue

            content_lower = doc.content.lower()
            content_words = set(content_lower.split())

            # Score based on word overlap
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words)
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(document=doc, score=score, rank=i+1)
            for i, (doc, score) in enumerate(results[:k])
        ]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._index_dirty = True
            if self.persist_path:
                self._save_to_disk()
            return True
        return False

    def update(self, doc_id: str, content: Optional[str] = None,
              metadata: Optional[Dict] = None) -> Optional[VectorDocument]:
        """Update a document."""
        doc = self.documents.get(doc_id)
        if not doc:
            return None

        if content is not None:
            doc.content = content
            doc.embedding = None  # Clear embedding for re-computation
            self._index_dirty = True

        if metadata is not None:
            doc.metadata.update(metadata)

        doc.updated_at = datetime.now()

        if self.persist_path:
            self._save_to_disk()

        return doc

    def list_documents(self, limit: int = 100,
                      filter_metadata: Optional[Dict] = None) -> List[VectorDocument]:
        """List documents with optional filtering."""
        docs = list(self.documents.values())

        if filter_metadata:
            docs = [d for d in docs if self._matches_filter(d.metadata, filter_metadata)]

        return docs[:limit]

    def count(self) -> int:
        """Get total document count."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self._index_dirty = True
        if self.persist_path:
            self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Save store to disk."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": {
                doc_id: doc.to_dict()
                for doc_id, doc in self.documents.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_from_disk(self) -> None:
        """Load store from disk."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.documents = {
                doc_id: VectorDocument.from_dict(doc_data)
                for doc_id, doc_data in data.get("documents", {}).items()
            }
        except Exception:
            pass  # Start with empty store on error

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        embedded_count = sum(1 for d in self.documents.values() if d.embedding is not None)

        return {
            "total_documents": len(self.documents),
            "embedded_documents": embedded_count,
            "persist_path": self.persist_path,
            "has_embedding_provider": self.embedding_provider is not None
        }
