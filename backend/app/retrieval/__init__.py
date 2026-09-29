"""Retrieval package for TrustRAG semantic search and vector indexing."""

from backend.app.retrieval.faiss_index import FaissIndexError, FaissVectorIndex

__all__ = ["FaissVectorIndex", "FaissIndexError"]
