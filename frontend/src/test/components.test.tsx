import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryInput } from '../components/QueryInput';
import { AnswerView } from '../components/AnswerView';
import { TrustDecision } from '../components/TrustDecision';
import { EvidencePanel } from '../components/EvidencePanel';
import { StatusMessage } from '../components/StatusMessage';
import { PipelineVisualization } from '../components/PipelineVisualization';
import type { QueryResponse, CitationItem } from '../types/trustrag';

describe('TrustRAG Component Tests', () => {
  it('Requirement 2 & 3: Query input works and empty query is prevented', () => {
    const handleSubmit = vi.fn();
    const handleReset = vi.fn();

    render(
      <QueryInput
        onSubmit={handleSubmit}
        onReset={handleReset}
        isLoading={false}
      />
    );

    const input = screen.getByTestId('query-input');
    const submitBtn = screen.getByTestId('query-submit-btn');

    // Initially empty -> submit button is disabled (Requirement 3)
    expect(submitBtn).toBeDisabled();

    // Type whitespace only -> still disabled
    fireEvent.change(input, { target: { value: '   ' } });
    expect(submitBtn).toBeDisabled();

    // Type valid query -> enabled (Requirement 2)
    fireEvent.change(input, { target: { value: 'What is a Data Principal?' } });
    expect(submitBtn).not.toBeDisabled();

    fireEvent.click(submitBtn);
    expect(handleSubmit).toHaveBeenCalledWith('What is a Data Principal?');
  });

  it('Requirement 5 & 6: Successful answer and citations render properly', () => {
    const mockCitation: CitationItem = {
      index: 1,
      chunk_id: 'chunk-abc',
      document_id: 'doc-1',
      document_name: 'DPDP_Act_2023.pdf',
      page_start: 3,
      page_end: 3,
      text_snippet: 'Data Principal means the individual to whom the personal data relates.',
      formatted_reference: '[1] DPDP_Act_2023.pdf, page 3',
    };

    const mockResponse: QueryResponse = {
      query: 'What is a Data Principal?',
      answer: 'A Data Principal is the individual to whom data relates [1].',
      decision: 'SUPPORTED',
      is_refusal: false,
      citations: [mockCitation],
      formatted_response: 'A Data Principal is the individual to whom data relates [1].',
      trust_metrics: {
        confidence_score: 0.96,
        groundedness_score: 0.99,
        relevance_score: 0.94,
        coverage_score: 1.0,
        provenance_valid: true,
      },
      latency_seconds: 0.85,
      trace_events: [],
      metadata: {},
    };

    render(<AnswerView response={mockResponse} />);

    // Answer text renders
    expect(screen.getByTestId('answer-text')).toHaveTextContent('A Data Principal is the individual');
    // Citation tag rendered
    expect(screen.getByText('[1]')).toBeInTheDocument();
    // Citation card rendered
    expect(screen.getByTestId('citation-card-1')).toHaveTextContent('DPDP_Act_2023.pdf');
    expect(screen.getByTestId('citation-card-1')).toHaveTextContent('Page 3');
  });

  it('Requirement 7: Supported trust state renders with high confidence', () => {
    render(
      <TrustDecision
        decision="SUPPORTED"
        isRefusal={false}
        metrics={{
          confidence_score: 0.92,
          groundedness_score: 0.95,
          relevance_score: 0.88,
          coverage_score: 0.9,
          provenance_valid: true,
        }}
        latencySeconds={1.1}
      />
    );

    const banner = screen.getByTestId('trust-banner');
    expect(banner).toHaveTextContent('SUPPORTED');
    expect(screen.getByTestId('confidence-score')).toHaveTextContent('92%');
    expect(screen.getByTestId('groundedness-score')).toHaveTextContent('95%');
    expect(screen.getByTestId('provenance-valid')).toHaveTextContent('Valid');
  });

  it('Requirement 8: Insufficient evidence refusal renders clearly without error failure', () => {
    const mockRefusalResponse: QueryResponse = {
      query: 'Who is the President of Mars?',
      answer: 'TrustRAG could not verify the answer from the available evidence.',
      decision: 'INSUFFICIENT_EVIDENCE',
      is_refusal: true,
      citations: [],
      formatted_response: 'TrustRAG could not verify the answer from the available evidence.',
      trust_metrics: {
        confidence_score: 0.12,
        groundedness_score: 0.05,
        relevance_score: 0.1,
        coverage_score: 0.0,
        provenance_valid: false,
      },
      latency_seconds: 0.45,
      trace_events: [],
      metadata: {},
    };

    render(<AnswerView response={mockRefusalResponse} />);
    expect(screen.getByTestId('refusal-notice')).toBeInTheDocument();
    expect(screen.getByTestId('refusal-notice')).toHaveTextContent('Insufficient Grounding Evidence');

    render(
      <TrustDecision
        decision="INSUFFICIENT_EVIDENCE"
        isRefusal={true}
        metrics={mockRefusalResponse.trust_metrics}
      />
    );
    expect(screen.getByTestId('trust-banner')).toHaveTextContent('INSUFFICIENT EVIDENCE');
  });

  it('Requirement 9: API Error and Network Error states render user-friendly messages', () => {
    const { rerender } = render(
      <StatusMessage
        status="API_ERROR"
        errorMessage="Invalid parameters submitted to TrustRAG"
      />
    );
    expect(screen.getByTestId('api-error-state')).toHaveTextContent('Invalid parameters submitted');

    rerender(
      <StatusMessage
        status="NETWORK_ERROR"
        errorMessage="Unable to reach backend"
      />
    );
    expect(screen.getByTestId('network-error-state')).toHaveTextContent('Backend Connection Unavailable');
  });

  it('Requirement 10: Loading state renders pipeline indicator', () => {
    render(<StatusMessage status="LOADING" errorMessage={null} />);
    expect(screen.getByTestId('loading-state')).toBeInTheDocument();
    expect(screen.getByText(/Retrieval/)).toBeInTheDocument();
  });

  it('Pipeline visualization renders 5 specialized agent steps', () => {
    render(
      <PipelineVisualization
        status="SUCCESS"
        traceEvents={[
          { agent_name: 'RetrievalAgent', status: 'COMPLETED', latency_seconds: 0.1 },
          { agent_name: 'EvidenceAgent', status: 'COMPLETED', latency_seconds: 0.2 },
          { agent_name: 'TrustAgent', status: 'COMPLETED', latency_seconds: 0.1 },
          { agent_name: 'GenerationAgent', status: 'COMPLETED', latency_seconds: 0.4 },
          { agent_name: 'CitationAgent', status: 'COMPLETED', latency_seconds: 0.05 },
        ]}
      />
    );

    expect(screen.getByTestId('pipeline-step-retrieve')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-step-evidence')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-step-trust')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-step-generate')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-step-cite')).toBeInTheDocument();
  });

  it('Evidence panel allows toggling between evidence snippets and trace events', () => {
    const mockCitation: CitationItem = {
      index: 1,
      chunk_id: 'chk-001',
      document_id: 'doc-001',
      document_name: 'Sample.pdf',
      page_start: 1,
      page_end: 1,
      text_snippet: 'Verified text from document',
      formatted_reference: '[1] Sample.pdf, page 1',
    };

    render(
      <EvidencePanel
        citations={[mockCitation]}
        traceEvents={[
          { agent_name: 'RetrievalAgent', status: 'COMPLETED', latency_seconds: 0.12 },
        ]}
      />
    );

    // Initial Evidence tab
    expect(screen.getByTestId('evidence-item-1')).toHaveTextContent('Sample.pdf');

    // Click Trace tab
    fireEvent.click(screen.getByTestId('tab-trace'));
    expect(screen.getByTestId('trace-list')).toHaveTextContent('RetrievalAgent');
  });
});
