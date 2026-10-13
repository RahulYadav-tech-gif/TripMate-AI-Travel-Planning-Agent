"""ChromaDB vector store and embedding interface for destination guides."""

from pathlib import Path
import logging
from typing import List, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from src.config import Settings, get_settings
from src.rag.ingest import DocumentChunk

logger = logging.getLogger(__name__)


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Embeds text using Gemini embed_content, falling back to local embeddings if needed."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dimension = settings.embedding_dimension
        self._genai_client = None
        self._local_ef = embedding_functions.DefaultEmbeddingFunction()

        if self.settings.validate_api_key():
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.settings.gemini_api_key)
                logger.info("Initialized Gemini Client for embeddings (%s)", settings.gemini_embedding_model)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini Client: %s. Using local embeddings.", exc)

    @staticmethod
    def name() -> str:
        return "gemini_embedding_function"

    def get_config(self) -> dict:
        return {"dimension": self.dimension}

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(get_settings())

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Computes embeddings for a list of strings."""
        if self._genai_client is not None:
            try:
                from google.genai import types
                response = self._genai_client.models.embed_content(
                    model=self.settings.gemini_embedding_model,
                    contents=input,
                    config=types.EmbedContentConfig(output_dimensionality=self.dimension),
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    return [list(emb.values) for emb in response.embeddings]
            except Exception as exc:
                logger.warning("Gemini embedding API call failed: %s. Using local embeddings.", exc)

        return self._local_ef(input)


class DestinationVectorStore:
    """Manages the ChromaDB collection for destination guides."""

    COLLECTION_NAME = "destination_guides"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_db_dir)
        self.embedding_fn = GeminiEmbeddingFunction(self.settings)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_chunks(self, chunks: List[DocumentChunk], force_reload: bool = False) -> int:
        """Adds document chunks into the ChromaDB collection."""
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reload:
            logger.info("Collection '%s' already has %d documents. Skipping.", self.COLLECTION_NAME, existing_count)
            return existing_count

        if force_reload and existing_count > 0:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Ingested %d chunks into '%s'.", len(ids), self.COLLECTION_NAME)
        return self.collection.count()

    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Runs cosine similarity search on the collection and returns text chunks."""
        k = top_k if top_k is not None else self.settings.rag_top_k
        query_clean = query.strip()
        if not query_clean or self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_clean],
                n_results=min(k, self.collection.count()),
            )
        except Exception as exc:
            # Handle dimension changes between local ONNX (384) and Gemini (768)
            if "dimension" in str(exc).lower():
                logger.warning("Dimension mismatch detected in ChromaDB (%s). Re-indexing...", exc)
                from src.rag.ingest import load_all_destination_chunks
                chunks = load_all_destination_chunks(Path(self.settings.data_dir))
                self.ingest_chunks(chunks, force_reload=True)
                results = self.collection.query(
                    query_texts=[query_clean],
                    n_results=min(k, self.collection.count()),
                )
            else:
                logger.error("ChromaDB query failed: %s", exc)
                return []

        documents = results.get("documents", [])
        if documents and documents[0]:
            return documents[0]
        return []
