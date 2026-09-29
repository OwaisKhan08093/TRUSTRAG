"""Metadata resolution layer mapping FAISS vector indices back to chunk documents."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.config import CHUNKS_OUTPUT_FILE, EMBEDDING_METADATA_FILE


class MetadataResolutionError(Exception):
    """Raised when metadata lookup, mapping, or indexing fails."""
    pass


class ChunkMetadataResolver:
    """Resolves FAISS index integer positions to complete document chunk metadata."""

    def __init__(
        self,
        chunks_file: Union[str, Path] = CHUNKS_OUTPUT_FILE,
        metadata_file: Union[str, Path] = EMBEDDING_METADATA_FILE,
    ) -> None:
        """Load and index chunk metadata and embedding index mappings.

        Args:
            chunks_file: Path to processed chunks.json file.
            metadata_file: Path to embedding_metadata.json mapping file.

        Raises:
            FileNotFoundError: If chunks_file or metadata_file does not exist.
            MetadataResolutionError: If files are corrupted, have duplicate IDs, or have broken references.
        """
        self._chunks_path = Path(chunks_file)
        self._metadata_path = Path(metadata_file)

        self._index_to_chunk_id: Dict[int, str] = {}
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}

        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Load JSON files and build internal lookup tables with rigorous invariant validation."""
        if not self._chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at '{self._chunks_path}'.")

        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Embedding metadata file not found at '{self._metadata_path}'.")

        try:
            with open(self._chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
        except Exception as exc:
            raise MetadataResolutionError(f"Failed to read chunks JSON from '{self._chunks_path}': {exc}") from exc

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
        except Exception as exc:
            raise MetadataResolutionError(
                f"Failed to read embedding metadata JSON from '{self._metadata_path}': {exc}"
            ) from exc

        if not isinstance(raw_chunks, list):
            raise MetadataResolutionError(f"Chunks file must contain a list of chunk objects, got {type(raw_chunks).__name__}.")

        if not isinstance(raw_metadata, list):
            raise MetadataResolutionError(
                f"Embedding metadata must contain a list of mapping objects, got {type(raw_metadata).__name__}."
            )

        # 1. Build chunk lookup table and check for duplicates
        self._chunks_by_id.clear()
        for idx, chunk in enumerate(raw_chunks):
            if not isinstance(chunk, dict):
                raise MetadataResolutionError(f"Chunk at position {idx} is not a valid JSON object.")

            chunk_id = chunk.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise MetadataResolutionError(f"Chunk at position {idx} is missing a valid 'chunk_id'.")

            if chunk_id in self._chunks_by_id:
                raise MetadataResolutionError(f"Duplicate chunk_id '{chunk_id}' detected in chunks file.")

            # Validate required schema fields
            required_fields = ["document_id", "document_name", "page_start", "page_end", "text"]
            for field in required_fields:
                if field not in chunk:
                    raise MetadataResolutionError(f"Chunk '{chunk_id}' is missing required field '{field}'.")

            self._chunks_by_id[chunk_id] = {
                "chunk_id": str(chunk_id),
                "document_id": str(chunk["document_id"]),
                "document_name": str(chunk["document_name"]),
                "page_start": int(chunk["page_start"]),
                "page_end": int(chunk["page_end"]),
                "text": str(chunk["text"]),
            }

        # 2. Build index -> chunk_id table
        self._index_to_chunk_id.clear()
        for idx, entry in enumerate(raw_metadata):
            if not isinstance(entry, dict):
                raise MetadataResolutionError(f"Metadata entry at position {idx} is not a dictionary.")

            emb_idx = entry.get("embedding_index")
            chunk_id = entry.get("chunk_id")

            if emb_idx is None or not isinstance(emb_idx, int) or emb_idx < 0:
                raise MetadataResolutionError(f"Metadata entry at position {idx} has invalid 'embedding_index': {emb_idx}.")

            if not chunk_id or not isinstance(chunk_id, str):
                raise MetadataResolutionError(f"Metadata entry at position {idx} has invalid 'chunk_id': {chunk_id}.")

            if emb_idx in self._index_to_chunk_id:
                raise MetadataResolutionError(f"Duplicate embedding index {emb_idx} found in metadata mapping.")

            if chunk_id not in self._chunks_by_id:
                raise MetadataResolutionError(
                    f"Metadata references chunk_id '{chunk_id}' (index {emb_idx}) which does not exist in chunks file."
                )

            self._index_to_chunk_id[emb_idx] = chunk_id

        # 3. Ensure continuity of indices (0 to N-1)
        expected_indices = set(range(len(raw_metadata)))
        actual_indices = set(self._index_to_chunk_id.keys())
        if expected_indices != actual_indices:
            raise MetadataResolutionError(
                f"Embedding indices are non-contiguous. Missing indices: {expected_indices - actual_indices}"
            )

    @property
    def total_chunks(self) -> int:
        """Return the total number of registered chunks."""
        return len(self._chunks_by_id)

    @property
    def total_mappings(self) -> int:
        """Return the total number of embedding index mappings."""
        return len(self._index_to_chunk_id)

    def resolve_index(self, index: int) -> Dict[str, Any]:
        """Resolve a single FAISS index position to its complete chunk metadata.

        Args:
            index: Non-negative integer index from FAISS.

        Returns:
            Dictionary with chunk_id, document_id, document_name, page_start, page_end, text.

        Raises:
            TypeError: If index is not an integer.
            IndexError: If index is out of bounds or unmapped.
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index).__name__}.")

        if index not in self._index_to_chunk_id:
            raise IndexError(
                f"Index {index} is out of bounds. Available indices range from 0 to {len(self._index_to_chunk_id) - 1}."
            )

        chunk_id = self._index_to_chunk_id[index]
        return dict(self._chunks_by_id[chunk_id])

    def resolve_indices(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        """Resolve a sequence of FAISS index positions preserving order.

        Args:
            indices: Sequence of integer indices.

        Returns:
            List of resolved chunk metadata dictionaries.
        """
        return [self.resolve_index(idx) for idx in indices]
