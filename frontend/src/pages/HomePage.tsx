import React, { useState } from 'react';
import { DocumentUpload } from '../components/DocumentUpload';
import { QueryInput } from '../components/QueryInput';
import { StatusMessage } from '../components/StatusMessage';
import { PipelineVisualization } from '../components/PipelineVisualization';
import { TrustDecision } from '../components/TrustDecision';
import { AnswerView } from '../components/AnswerView';
import { EvidencePanel } from '../components/EvidencePanel';
import type { UseTrustRagReturn } from '../hooks/useTrustRag';
import type { CitationItem } from '../types/trustrag';

interface HomePageProps {
  trustRag: UseTrustRagReturn;
}

export const HomePage: React.FC<HomePageProps> = ({ trustRag }) => {
  const {
    status,
    response,
    errorMessage,
    currentStepMessage,
    documents,
    isUploading,
    uploadSuccessMessage,
    uploadErrorMessage,
    uploadDocument,
    submitQuery,
    reset,
  } = trustRag;

  const [selectedCitation, setSelectedCitation] = useState<CitationItem | null>(null);

  const handleSelectCitation = (citation: CitationItem) => {
    setSelectedCitation(citation);
  };

  const handleReset = () => {
    setSelectedCitation(null);
    reset();
  };

  const hasResult = response !== null && (status === 'SUCCESS' || status === 'INSUFFICIENT_EVIDENCE');

  return (
    <div className="main-grid" data-testid="home-page">
      {/* Document Ingestion & Catalog Section */}
      <DocumentUpload
        documents={documents}
        isUploading={isUploading}
        uploadSuccessMessage={uploadSuccessMessage}
        uploadErrorMessage={uploadErrorMessage}
        onUpload={uploadDocument}
      />

      {/* Query Input Box */}
      <QueryInput
        onSubmit={(query) => submitQuery(query)}
        onReset={handleReset}
        isLoading={status === 'LOADING'}
      />

      {/* Loading / Error States */}
      <StatusMessage
        status={status}
        errorMessage={errorMessage}
        loadingMessage={currentStepMessage}
        onRetry={() => {
          if (response?.query) {
            submitQuery(response.query);
          }
        }}
      />

      {/* Visual Pipeline Flow */}
      <PipelineVisualization
        status={status}
        traceEvents={response?.trace_events}
        decision={response?.decision}
        isRefusal={response?.is_refusal}
      />

      {/* Main Results View */}
      {hasResult && response && (
        <div className="results-layout" data-testid="results-layout">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <TrustDecision
              decision={response.decision}
              isRefusal={response.is_refusal}
              metrics={response.trust_metrics}
              latencySeconds={response.latency_seconds}
            />

            <AnswerView
              response={response}
              onCitationClick={handleSelectCitation}
              selectedCitationIndex={selectedCitation?.index}
            />
          </div>

          <div style={{ position: 'sticky', top: '1.5rem' }}>
            <EvidencePanel
              citations={response.citations || []}
              traceEvents={response.trace_events || []}
              metadata={response.metadata || {}}
              activeCitationIndex={selectedCitation?.index}
              onSelectCitation={handleSelectCitation}
            />
          </div>
        </div>
      )}
    </div>
  );
};
