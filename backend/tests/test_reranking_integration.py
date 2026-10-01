"""Comprehensive integration tests for end-to-end Cross-Encoder Reranking Subsystem (Milestone 9)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    MAX_RERANKER_TOP_K,
)
from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.reranking.config import RerankerConfig, RerankerConfigurationError
from backend.app.reranking.cross_encoder import CrossEncoderModelError, CrossEncoderReranker
from backend.app.reranking.pipeline import RerankingPipeline, RerankingPipelineError
from backend.app.reranking.reranker import ResultReranker, RerankerError
from backend.app.reranking.schema import RerankedChunk, SchemaValidationError
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.faiss_index import FaissVectorIndex
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.metadata import ChunkMetadataResolver
from backend.app.retrieval.retriever import VectorRetriever


@pytest.fixture(scope="module")
def shared_encoder() -> EmbeddingEncoder:
    """Shared EmbeddingEncoder fixture."""
    return EmbeddingEncoder()


@pytest.fixture(scope="module")
def shared_cross_encoder() -> CrossEncoderReranker:
    """Shared CrossEncoderReranker fixture using default model."""
    return CrossEncoderReranker()


@pytest.fixture
def integrated_environment(tmp_path: Path, shared_encoder: EmbeddingEncoder, shared_cross_encoder: CrossEncoderReranker):
    """Set up an isolated end-to-end environment with chunks, FAISS index, BM25, and HybridRetriever."""
    chunks = [
        {
            "chunk_id": "dpdp_sec1",
            "document_id": "dpdp_2023",
            "document_name": "dpdp_act_2023.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "The Digital Personal Data Protection Act 2023 provides for processing of digital personal data.",
        },
        {
            "chunk_id": "dpdp_sec4",
            "document_id": "dpdp_2023",
            "document_name": "dpdp_act_2023.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "A person may process the personal data of a Data Principal only in accordance with the provisions of this Act.",
        },
        {
            "chunk_id": "dpdp_sec13",
            "document_id": "dpdp_2023",
            "document_name": "dpdp_act_2023.pdf",
            "page_start": 5,
            "page_end": 6,
            "text": "Right to grievance redressal: A Data Principal shall have the right to register grievance with the Data Fiduciary.",
        },
    ]

    chunks_file = tmp_path / "chunks.json"
    meta_file = tmp_path / "embedding_metadata.json"
    faiss_file = tmp_path / "index.faiss"

    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

    # Build embeddings and FAISS index
    texts = [c["text"] for c in chunks]
    embeddings = shared_encoder.encode_documents(texts, normalize_embeddings=True)
    faiss_idx = FaissVectorIndex(dimension=embeddings.shape[1])
    faiss_idx.add_embeddings(embeddings)
    faiss_idx.save(faiss_file)

    meta_entries = [{"embedding_index": i, "chunk_id": chunks[i]["chunk_id"]} for i in range(len(chunks))]
    meta_file.write_text(json.dumps(meta_entries), encoding="utf-8")

    resolver = ChunkMetadataResolver(chunks_file=chunks_file, metadata_file=meta_file)
    dense_retriever = VectorRetriever(index=faiss_idx, encoder=shared_encoder, metadata_resolver=resolver)
    bm25_idx = BM25Index.from_chunks_file(chunks_file)
    sparse_retriever = BM25Retriever(index=bm25_idx, metadata_resolver=resolver)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        default_top_k=3,
        rrf_k=60,
    )

    reranker = ResultReranker(
        cross_encoder=shared_cross_encoder,
        default_top_k=2,
        default_batch_size=DEFAULT_RERANKER_BATCH_SIZE,
    )

    pipeline = RerankingPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        default_candidate_k=3,
        default_final_k=2,
    )

    return {
        "pipeline": pipeline,
        "hybrid": hybrid_retriever,
        "reranker": reranker,
        "chunks": chunks,
    }


def test_end_to_end_reranking_flow(integrated_environment):
    """Verify complete flow: Query -> HybridRetriever -> RerankingPipeline -> CrossEncoder -> Reranked Evidence."""
    pipeline = integrated_environment["pipeline"]
    query = "How can a Data Principal seek grievance redressal?"

    results = pipeline.retrieve(query=query, top_k=2, candidate_k=3)

    assert len(results) == 2
    assert isinstance(results[0], RerankedChunk)
    assert isinstance(results[1], RerankedChunk)

    # Top match should be sec13 regarding grievance redressal
    assert results[0].chunk_id == "dpdp_sec13"
    assert results[0].final_rank == 1
    assert results[1].final_rank == 2

    # Scores should be in descending order
    assert results[0].rerank_score >= results[1].rerank_score

    # Provenance fields must be fully preserved
    assert results[0].document_id == "dpdp_2023"
    assert results[0].document_name == "dpdp_act_2023.pdf"
    assert results[0].page_start == 5
    assert results[0].page_end == 6
    assert "grievance" in results[0].text.lower()
    assert results[0].original_rank >= 1
    assert isinstance(results[0].original_score, float)
    assert isinstance(results[0].rerank_score, float)


def test_edge_case_empty_query(integrated_environment):
    """Verify pipeline rejects empty strings or whitespace queries."""
    pipeline = integrated_environment["pipeline"]

    with pytest.raises(ValueError, match="query cannot be empty"):
        pipeline.retrieve("")

    with pytest.raises(ValueError, match="query cannot be empty"):
        pipeline.retrieve("   \t\n  ")

    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.retrieve(None)  # type: ignore


def test_edge_case_empty_candidates(integrated_environment):
    """Verify pipeline handles empty retriever candidates gracefully."""
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = []
    reranker = integrated_environment["reranker"]

    pipeline = RerankingPipeline(retriever=mock_retriever, reranker=reranker)
    results = pipeline.retrieve("any query")

    assert results == []


def test_edge_case_invalid_top_k(integrated_environment):
    """Verify pipeline rejects invalid top_k and final_k boundaries."""
    pipeline = integrated_environment["pipeline"]

    with pytest.raises(ValueError, match="final_k must be a positive integer"):
        pipeline.retrieve("query", top_k=0)

    with pytest.raises(ValueError, match="final_k must be a positive integer"):
        pipeline.retrieve("query", top_k=-1)

    with pytest.raises(ValueError, match="cannot exceed maximum limit"):
        pipeline.retrieve("query", top_k=MAX_RERANKER_TOP_K + 1, candidate_k=MAX_RERANKER_TOP_K + 1)


def test_edge_case_candidate_k_greater_than_corpus(integrated_environment):
    """Verify candidate_k larger than corpus returns all available scored candidates."""
    pipeline = integrated_environment["pipeline"]
    # Corpus has 3 documents, candidate_k=10, final_k=3
    results = pipeline.retrieve("Digital Personal Data Protection Act", candidate_k=10, top_k=3)

    assert len(results) == 3
    for i, res in enumerate(results, start=1):
        assert res.final_rank == i


def test_edge_case_final_k_greater_than_candidate_k(integrated_environment):
    """Verify pipeline raises ValueError when final_k > candidate_k."""
    pipeline = integrated_environment["pipeline"]

    with pytest.raises(ValueError, match="cannot exceed candidate_k"):
        pipeline.retrieve("query", top_k=5, candidate_k=2)


def test_edge_case_invalid_model():
    """Verify CrossEncoderReranker raises CrossEncoderModelError when loading an invalid model name."""
    with pytest.raises(CrossEncoderModelError):
        CrossEncoderReranker(model_name="non_existent_hf_model_path_12345_xyz")


def test_edge_case_malformed_results(integrated_environment):
    """Verify ResultReranker raises SchemaValidationError or RerankerError on malformed candidates."""
    reranker = integrated_environment["reranker"]

    # Candidate missing 'text'
    malformed = [
        {"chunk_id": "c1", "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1}
    ]
    with pytest.raises(ValueError, match="missing a valid string 'text' key"):
        reranker.rerank(query="query", results=malformed)

    # Candidate missing 'chunk_id' in schema generation
    malformed_schema = [
        {"text": "some text", "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1}
    ]
    with pytest.raises(RerankerError, match="Failed to construct RerankedChunk"):
        reranker.rerank(query="query", results=malformed_schema)


def test_edge_case_duplicate_chunks(integrated_environment):
    """Verify ResultReranker deterministically orders duplicate candidates."""
    reranker = integrated_environment["reranker"]

    duplicates = [
        {"chunk_id": "dup_1", "text": "Same text content", "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1, "rank": 1, "score": 0.05},
        {"chunk_id": "dup_1", "text": "Same text content", "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1, "rank": 2, "score": 0.04},
    ]

    results = reranker.rerank("query", duplicates, top_k=2)
    assert len(results) == 2
    assert results[0].final_rank == 1
    assert results[1].final_rank == 2


def test_edge_case_missing_text(integrated_environment):
    """Verify ResultReranker rejects candidates with non-string text."""
    reranker = integrated_environment["reranker"]

    invalid_text_cand = [
        {"chunk_id": "c1", "text": 12345, "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1, "rank": 1, "score": 0.05}
    ]
    with pytest.raises(ValueError, match="missing a valid string 'text' key"):
        reranker.rerank(query="query", results=invalid_text_cand)  # type: ignore


def test_edge_case_score_mismatch_and_nan(integrated_environment):
    """Verify RerankedChunk validation detects NaN or non-finite scores."""
    with pytest.raises(SchemaValidationError, match="rerank_score must be a finite float"):
        RerankedChunk(
            chunk_id="c1",
            document_id="d1",
            document_name="d.pdf",
            page_start=1,
            page_end=1,
            text="text",
            original_rank=1,
            original_score=0.5,
            rerank_score=float("nan"),
            final_rank=1,
        )


def test_provenance_preservation_dictionary_conversion(integrated_environment):
    """Verify to_dict and from_dict preserve all 10 standard schema fields."""
    pipeline = integrated_environment["pipeline"]
    results = pipeline.retrieve("Digital Personal Data Protection Act", top_k=1)
    assert len(results) == 1
    chunk = results[0]

    d = chunk.to_dict()
    assert set(d.keys()) == {
        "chunk_id",
        "document_id",
        "document_name",
        "page_start",
        "page_end",
        "text",
        "original_rank",
        "original_score",
        "rerank_score",
        "final_rank",
    }

    reconstructed = RerankedChunk.from_dict(d)
    assert reconstructed == chunk


def test_batch_size_variations(integrated_environment):
    """Verify pipeline functions identically across different batch sizes."""
    pipeline = integrated_environment["pipeline"]
    query = "Data Principal rights"

    res_b1 = pipeline.retrieve(query, candidate_k=3, top_k=3, batch_size=1)
    res_b16 = pipeline.retrieve(query, candidate_k=3, top_k=3, batch_size=16)

    assert len(res_b1) == len(res_b16) == 3
    for c1, c16 in zip(res_b1, res_b16):
        assert c1.chunk_id == c16.chunk_id
        assert c1.final_rank == c16.final_rank
        assert pytest.approx(c1.rerank_score, rel=1e-4) == c16.rerank_score
