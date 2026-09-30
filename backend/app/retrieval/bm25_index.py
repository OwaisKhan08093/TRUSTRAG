"""BM25 sparse retrieval index for lexical keyword matching."""

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from rank_bm25 import BM25Okapi

from backend.app.config import CHUNKS_OUTPUT_FILE


class BM25IndexError(Exception):
    """Raised when BM25 indexing or searching fails."""
    pass


def default_tokenizer(text: str) -> List[str]:
    """Tokenize input text into lowercase alphanumeric tokens.

    Args:
        text: Input string to tokenize.

    Returns:
        List of lowercase token strings.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Sparse inverted index using BM25Okapi for keyword-based retrieval."""

    def __init__(
        self,
        chunks: Sequence[Union[Dict[str, Any], str]],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """Initialize and build a BM25 index from a sequence of chunks.

        Args:
            chunks: Sequence of chunk dictionaries (with 'text' key) or raw text strings.
            tokenizer: Optional custom tokenizer function mapping str -> List[str].

        Raises:
            TypeError: If chunks is not a sequence or contains invalid types.
            ValueError: If chunks is empty.
        """
        if not isinstance(chunks, (list, tuple)):
            raise TypeError(f"Chunks must be a list or tuple, got {type(chunks).__name__}.")

        if len(chunks) == 0:
            raise ValueError("Corpus cannot be empty. At least one document chunk is required.")

        self._tokenizer = tokenizer or default_tokenizer
        self._raw_chunks: List[Union[Dict[str, Any], str]] = []
        self._corpus_texts: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

        self._build_index(chunks)

    def _build_index(self, chunks: Sequence[Union[Dict[str, Any], str]]) -> None:
        """Extract text, tokenize, and initialize BM25Okapi data structures."""
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                if "text" not in chunk:
                    raise ValueError(f"Chunk at index {idx} is a dictionary missing required 'text' key.")
                text_content = chunk["text"]
                if not isinstance(text_content, str):
                    raise TypeError(f"Chunk 'text' at index {idx} must be a string, got {type(text_content).__name__}.")
                self._raw_chunks.append(dict(chunk))
                self._corpus_texts.append(text_content)
            elif isinstance(chunk, str):
                self._raw_chunks.append(chunk)
                self._corpus_texts.append(chunk)
            else:
                raise TypeError(
                    f"Chunk at index {idx} must be a dict or string, got {type(chunk).__name__}."
                )

            tokens = self._tokenizer(self._corpus_texts[-1])
            self._tokenized_corpus.append(tokens)

        try:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as exc:
            raise BM25IndexError(f"Failed to build BM25Okapi index: {exc}") from exc

    @property
    def corpus_size(self) -> int:
        """Return the number of document chunks indexed in the BM25 corpus."""
        return len(self._corpus_texts)

    @property
    def tokenized_corpus(self) -> List[List[str]]:
        """Return the tokenized representation of the indexed corpus."""
        return list(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the BM25 index using lexical keyword matching.

        Args:
            query: Non-empty query string.
            top_k: Maximum number of top ranked results to return (positive integer).

        Returns:
            List of dictionaries with 'index' (0-based chunk integer position)
            and 'score' (BM25 float score), sorted in descending order of score.
            Never returns more results than the corpus size.

        Raises:
            TypeError: If query is not a string or top_k is not an integer.
            ValueError: If query is empty/whitespace or top_k < 1.
            BM25IndexError: If internal BM25 scoring fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query string cannot be empty or contain only whitespace.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k}.")

        # Tokenize query using index's tokenizer
        query_tokens = self._tokenizer(stripped_query)

        try:
            # Compute BM25 scores for all corpus documents
            scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:
            raise BM25IndexError(f"BM25 scoring failed: {exc}") from exc

        # Determine effective top_k bounded by corpus size
        effective_k = min(top_k, self.corpus_size)

        # Sort indices by score descending; break ties by original document order
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        results: List[Dict[str, Any]] = [
            {"index": idx, "score": score}
            for idx, score in indexed_scores[:effective_k]
        ]
        return results

    @classmethod
    def from_chunks_file(
        cls,
        chunks_file: Union[str, Path] = CHUNKS_OUTPUT_FILE,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Index":
        """Load document chunks from a JSON file and construct a BM25Index.

        Args:
            chunks_file: Path to processed chunks.json file.
            tokenizer: Optional custom tokenizer function.

        Returns:
            Instantiated and indexed BM25Index.

        Raises:
            FileNotFoundError: If chunks_file does not exist.
            ValueError: If file is empty or contains non-list data.
        """
        path = Path(chunks_file)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found at '{path}'.")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise BM25IndexError(f"Failed to parse JSON from '{path}': {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Chunks file must contain a JSON list, got {type(data).__name__}.")

        return cls(chunks=data, tokenizer=tokenizer)
