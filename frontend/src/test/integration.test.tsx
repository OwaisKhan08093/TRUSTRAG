import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import type { QueryResponse, HealthResponse } from '../types/trustrag';

describe('TrustRAG Full Frontend Integration', () => {
  const originalFetch = globalThis.fetch;

  const mockHealth: HealthResponse = {
    status: 'healthy',
    version: '1.0.0',
    device: 'cpu',
    pipeline_ready: true,
  };

  const mockQueryResponse: QueryResponse = {
    query: 'What is personal data?',
    answer: 'Personal data is any data about an individual [1].',
    decision: 'SUPPORTED',
    is_refusal: false,
    citations: [
      {
        index: 1,
        chunk_id: 'chunk-999',
        document_id: 'doc-dpdp',
        document_name: 'DPDP_Act_2023.pdf',
        page_start: 2,
        page_end: 2,
        text_snippet: 'Personal data means any data about an individual who is identifiable.',
        formatted_reference: '[1] DPDP_Act_2023.pdf, page 2',
      },
    ],
    formatted_response: 'Personal data is any data about an individual [1].',
    trust_metrics: {
      confidence_score: 0.94,
      groundedness_score: 0.97,
      relevance_score: 0.91,
      coverage_score: 1.0,
      provenance_valid: true,
    },
    latency_seconds: 1.15,
    trace_events: [
      { agent_name: 'RetrievalAgent', status: 'COMPLETED', latency_seconds: 0.2 },
      { agent_name: 'EvidenceAgent', status: 'COMPLETED', latency_seconds: 0.3 },
      { agent_name: 'TrustAgent', status: 'COMPLETED', latency_seconds: 0.2 },
      { agent_name: 'GenerationAgent', status: 'COMPLETED', latency_seconds: 0.4 },
      { agent_name: 'CitationAgent', status: 'COMPLETED', latency_seconds: 0.05 },
    ],
    metadata: { session_id: 'sess-test' },
  };

  beforeEach(() => {
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes('/health')) {
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify(mockHealth),
        };
      }
      if (url.includes('/query')) {
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify(mockQueryResponse),
        };
      }
      return {
        ok: false,
        status: 404,
        text: async () => JSON.stringify({ detail: 'Not found' }),
      };
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('Requirement 1: Complete application renders shell, title, input, and visualization', async () => {
    render(<App />);

    expect(screen.getByText('TrustRAG')).toBeInTheDocument();
    expect(screen.getByTestId('query-input')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-visualization')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/API Online/)).toBeInTheDocument();
    });
  });


  it('Executes end-to-end query flow: input -> submit -> loading -> results view with trust metrics', async () => {
    render(<App />);

    const input = screen.getByTestId('query-input');
    const submitBtn = screen.getByTestId('query-submit-btn');

    fireEvent.change(input, { target: { value: 'What is personal data?' } });
    fireEvent.click(submitBtn);

    // Wait for async response to settle and verify rendered components
    await waitFor(() => {
      expect(screen.getByTestId('results-layout')).toBeInTheDocument();
    });

    expect(screen.getByTestId('answer-text')).toHaveTextContent('Personal data is any data about an individual');
    expect(screen.getByTestId('trust-banner')).toHaveTextContent('SUPPORTED');
    expect(screen.getByTestId('confidence-score')).toHaveTextContent('94%');
    expect(screen.getByTestId('evidence-panel')).toBeInTheDocument();
  });

  it('Handles query reset correctly', async () => {
    render(<App />);

    const input = screen.getByTestId('query-input');
    const submitBtn = screen.getByTestId('query-submit-btn');

    fireEvent.change(input, { target: { value: 'What is personal data?' } });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByTestId('results-layout')).toBeInTheDocument();
    });

    const resetBtn = screen.getByTestId('query-reset-btn');
    fireEvent.click(resetBtn);

    expect(screen.queryByTestId('results-layout')).not.toBeInTheDocument();
  });
});
