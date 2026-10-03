/**
 * TypeScript type definitions for TrustRAG API and frontend state.
 */

export interface QueryRequest {
  query: string;
  retrieval_top_k?: number;
  evidence_top_k?: number;
  max_new_tokens?: number;
  temperature?: number;
  filter_cited_only?: boolean;
  session_id?: string;
}

export interface CitationItem {
  index: number;
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_start: number;
  page_end: number;
  text_snippet: string;
  formatted_reference: string;
}

export interface TrustMetrics {
  confidence_score: number;
  groundedness_score: number;
  relevance_score: number;
  coverage_score: number;
  provenance_valid: boolean;
}

export interface TraceEventItem {
  agent_name: string;
  status: 'STARTED' | 'COMPLETED' | 'SKIPPED' | 'FAILED' | string;
  latency_seconds: number;
  reason?: string | null;
}

export interface QueryResponse {
  query: string;
  answer: string;
  decision: 'SUPPORTED' | 'INSUFFICIENT_EVIDENCE' | string;
  is_refusal: boolean;
  citations: CitationItem[];
  formatted_response: string;
  trust_metrics: TrustMetrics;
  latency_seconds: number;
  trace_events: TraceEventItem[];
  metadata: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  version: string;
  device?: string;
  pipeline_ready?: boolean;
  details?: Record<string, unknown>;
}

export type QueryStatus =
  | 'IDLE'
  | 'LOADING'
  | 'SUCCESS'
  | 'INSUFFICIENT_EVIDENCE'
  | 'API_ERROR'
  | 'NETWORK_ERROR';

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  chunks_created: number;
  total_chunks_indexed: number;
  message: string;
}

export interface DocumentInfo {
  document_id: string;
  document_name: string;
  chunks_count: number;
  pages_count: number;
  total_words: number;
  status: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total_documents: number;
  total_chunks: number;
}

