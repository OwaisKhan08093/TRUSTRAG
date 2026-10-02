import React from 'react';
import { AlertTriangle, WifiOff, Loader2, RefreshCw } from 'lucide-react';
import type { QueryStatus } from '../types/trustrag';

interface StatusMessageProps {
  status: QueryStatus;
  errorMessage: string | null;
  loadingMessage?: string;
  onRetry?: () => void;
}

export const StatusMessage: React.FC<StatusMessageProps> = ({
  status,
  errorMessage,
  loadingMessage = 'Processing query through TrustRAG pipeline...',
  onRetry,
}) => {
  if (status === 'LOADING') {
    return (
      <div className="glass-card loading-pulse" data-testid="loading-state">
        <div className="spinner" />
        <div>
          <div className="loading-text">{loadingMessage}</div>
          <div className="loading-subtext">
            Executing BM25 + FAISS Hybrid Retrieval &bull; Cross-Encoder Reranking &bull; TrustEngine Gating
          </div>
        </div>
      </div>
    );
  }

  if (status === 'NETWORK_ERROR') {
    return (
      <div
        className="glass-card"
        style={{
          borderColor: 'rgba(239, 68, 68, 0.4)',
          backgroundColor: 'var(--trust-error-bg)',
          padding: '1.5rem',
        }}
        data-testid="network-error-state"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <WifiOff size={22} color="#f87171" />
          <h3 style={{ fontSize: '1.05rem', color: '#f87171', fontWeight: 600 }}>
            Backend Connection Unavailable
          </h3>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
          {errorMessage || 'Could not connect to FastAPI server. Please ensure the backend is running on http://127.0.0.1:8000.'}
        </p>
        {onRetry && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onRetry}
            style={{ fontSize: '0.85rem', padding: '0.4rem 0.9rem' }}
          >
            <RefreshCw size={14} /> Retry Connection
          </button>
        )}
      </div>
    );
  }

  if (status === 'API_ERROR') {
    return (
      <div
        className="glass-card"
        style={{
          borderColor: 'rgba(239, 68, 68, 0.4)',
          backgroundColor: 'var(--trust-error-bg)',
          padding: '1.5rem',
        }}
        data-testid="api-error-state"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <AlertTriangle size={22} color="#f87171" />
          <h3 style={{ fontSize: '1.05rem', color: '#f87171', fontWeight: 600 }}>
            Query Processing Error
          </h3>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
          {errorMessage || 'The server encountered an error processing your query.'}
        </p>
        {onRetry && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onRetry}
            style={{ fontSize: '0.85rem', padding: '0.4rem 0.9rem' }}
          >
            <Loader2 size={14} /> Try Again
          </button>
        )}
      </div>
    );
  }

  return null;
};
