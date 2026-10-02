import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TrustRagApiClient, ApiError, NetworkError } from '../services/api';
import type { QueryRequest, QueryResponse, HealthResponse } from '../types/trustrag';

describe('TrustRagApiClient', () => {
  let client: TrustRagApiClient;
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    client = new TrustRagApiClient('http://localhost:8000', 5000);
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('Requirement 4: API client sends correct request to /query', async () => {
    const mockResponse: QueryResponse = {
      query: 'What is personal data?',
      answer: 'Personal data is any data about an individual [1].',
      decision: 'SUPPORTED',
      is_refusal: false,
      citations: [
        {
          index: 1,
          chunk_id: 'chunk-123',
          document_id: 'doc-1',
          document_name: 'DPDP_Act_2023.pdf',
          page_start: 2,
          page_end: 2,
          text_snippet: 'Personal data means any data about an individual.',
          formatted_reference: '[1] DPDP_Act_2023.pdf, page 2',
        },
      ],
      formatted_response: 'Personal data is any data about an individual [1].',
      trust_metrics: {
        confidence_score: 0.95,
        groundedness_score: 0.98,
        relevance_score: 0.92,
        coverage_score: 1.0,
        provenance_valid: true,
      },
      latency_seconds: 1.25,
      trace_events: [
        { agent_name: 'RetrievalAgent', status: 'COMPLETED', latency_seconds: 0.3 },
        { agent_name: 'EvidenceAgent', status: 'COMPLETED', latency_seconds: 0.4 },
        { agent_name: 'TrustAgent', status: 'COMPLETED', latency_seconds: 0.2 },
        { agent_name: 'GenerationAgent', status: 'COMPLETED', latency_seconds: 0.3 },
        { agent_name: 'CitationAgent', status: 'COMPLETED', latency_seconds: 0.05 },
      ],
      metadata: { session_id: 'sess-abc' },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(mockResponse),
    });

    const requestPayload: QueryRequest = {
      query: 'What is personal data?',
      filter_cited_only: true,
    };

    const result = await client.submitQuery(requestPayload);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/query',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify(requestPayload),
      })
    );
    expect(result.answer).toContain('Personal data is any data');
    expect(result.decision).toBe('SUPPORTED');
    expect(result.citations.length).toBe(1);
  });

  it('fetches health status from /health', async () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      version: '1.0.0',
      device: 'cpu',
      pipeline_ready: true,
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(mockHealth),
    });

    const result = await client.getHealth();
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/health',
      expect.objectContaining({ method: 'GET' })
    );
    expect(result.status).toBe('healthy');
  });

  it('Requirement 9: handles HTTP API errors appropriately', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      text: async () => JSON.stringify({ detail: 'query cannot be empty or whitespace only.' }),
    });

    await expect(client.submitQuery({ query: '   ' })).rejects.toThrow(ApiError);
  });

  it('handles network / fetch failure gracefully', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    await expect(client.submitQuery({ query: 'test' })).rejects.toThrow(NetworkError);
  });
});
