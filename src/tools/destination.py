"""Destination guide search tool (RAG).

Searches indexed travel guide documents for destination tips,
visa rules, local customs, and safety information.
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.config import get_settings
from src.rag.ingest import load_all_destination_chunks
from src.rag.store import DestinationVectorStore

logger = logging.getLogger(__name__)

# Cached vector store instance
_STORE_INSTANCE: Optional[DestinationVectorStore] = None


def get_vector_store() -> DestinationVectorStore:
    """Returns or initializes the DestinationVectorStore instance."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        settings = get_settings()
        _STORE_INSTANCE = DestinationVectorStore(settings)
        # Ingest docs if collection is empty
        if _STORE_INSTANCE.collection.count() == 0:
            chunks = load_all_destination_chunks(Path(settings.data_dir))
            _STORE_INSTANCE.ingest_chunks(chunks)
    return _STORE_INSTANCE


def search_destination_guide(query: str) -> List[str]:
    """Search the destination knowledge base for relevant travel guidance.

    Input: a natural-language question or topic.
    Output: the most relevant chunk(s) of text from the destination knowledge base.
    """
    if not query or not isinstance(query, str) or not query.strip():
        logger.warning("search_destination_guide called with empty query: %r", query)
        return []

    query_clean = query.strip()
    store = get_vector_store()
    results = store.search(query_clean)
    logger.info("search_destination_guide(%r) returned %d chunk(s)", query_clean, len(results))
    return results
