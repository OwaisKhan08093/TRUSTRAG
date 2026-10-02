"""Grounded query execution route for TrustRAG API."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.agents.base import AgentError, AgentInputError
from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.api.dependencies import get_orchestrator
from backend.app.api.schemas import (
    CitationItem,
    QueryRequest,
    QueryResponse,
    TraceEventItem,
    TrustMetrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


def map_orchestrator_result_to_response(result: OrchestratorResult) -> QueryResponse:
    """Map internal OrchestratorResult dataclass to public Pydantic QueryResponse schema."""
    citations_data: List[CitationItem] = []
    for c in result.citations:
        citations_data.append(
            CitationItem(
                index=c.index,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                page_start=c.page_start,
                page_end=c.page_end,
                text_snippet=c.text_snippet,
                formatted_reference=c.formatted_reference,
            )
        )

    metrics = TrustMetrics(
        confidence_score=result.confidence_score,
        groundedness_score=result.groundedness_score,
        relevance_score=result.relevance_score,
        coverage_score=result.coverage_score,
        provenance_valid=result.provenance_valid,
    )

    trace_items: List[TraceEventItem] = []
    if result.trace and result.trace.events:
        for evt in result.trace.events:
            trace_items.append(
                TraceEventItem(
                    agent_name=evt.agent_name,
                    status=evt.status.value,
                    latency_seconds=evt.latency_seconds,
                    reason=evt.reason,
                )
            )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        decision=result.decision.value,
        is_refusal=result.is_refusal,
        citations=citations_data,
        formatted_response=result.formatted_response,
        trust_metrics=metrics,
        latency_seconds=result.latency_seconds,
        trace_events=trace_items,
        metadata=result.metadata,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute Evidence-Grounded Query",
    response_description="Grounded response with citations, trust metrics, and execution trace.",
)
def execute_query(
    request: QueryRequest,
    orchestrator: TrustRAGOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """Execute evidence-constrained RAG query through multi-agent orchestration.

    The query pipeline coordinates:
    1. Hybrid dense + sparse retrieval
    2. Neural cross-encoder evidence reranking
    3. Multi-signal TrustEngine gating
    4. Grounded answer generation (if SUPPORTED)
    5. Citation reference compilation
    """
    try:
        orchestrator_result = orchestrator.execute(
            query=request.query,
            retrieval_top_k=request.retrieval_top_k,
            evidence_top_k=request.evidence_top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            filter_cited_only=request.filter_cited_only,
            session_id=request.session_id,
        )
    except AgentInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AgentError as exc:
        logger.error(f"Orchestrator runtime error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal orchestration failure: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected error during query execution: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error.",
        ) from exc

    return map_orchestrator_result_to_response(orchestrator_result)
