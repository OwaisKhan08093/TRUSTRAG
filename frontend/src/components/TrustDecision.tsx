import React from 'react';
import { ShieldCheck, ShieldAlert, CheckCircle2, XCircle, BarChart3, HelpCircle } from 'lucide-react';
import type { TrustMetrics } from '../types/trustrag';

interface TrustDecisionProps {
  decision: string;
  isRefusal: boolean;
  metrics?: TrustMetrics;
  latencySeconds?: number;
}

export const TrustDecision: React.FC<TrustDecisionProps> = ({
  decision,
  isRefusal,
  metrics,
  latencySeconds,
}) => {
  const isSupported = decision === 'SUPPORTED' && !isRefusal;

  const formatPercentage = (val?: number) => {
    if (typeof val !== 'number' || isNaN(val)) return 'N/A';
    return `${Math.round(val * 100)}%`;
  };

  return (
    <div className="glass-card" data-testid="trust-decision-container">
      <div className={`trust-banner ${isSupported ? 'supported' : 'refusal'}`} data-testid="trust-banner">
        <div>
          <div className="trust-status-title">
            {isSupported ? (
              <>
                <ShieldCheck size={24} color="#10b981" />
                <span>Deterministic Trust Status: SUPPORTED</span>
              </>
            ) : (
              <>
                <ShieldAlert size={24} color="#f59e0b" />
                <span>Deterministic Trust Status: INSUFFICIENT EVIDENCE</span>
              </>
            )}
          </div>
          <p className="trust-status-desc">
            {isSupported
              ? 'Response is verified, fully grounded in source passages, and backed by verifiable citations.'
              : 'TrustRAG could not verify the answer from the available evidence.'}
          </p>
        </div>

        {typeof latencySeconds === 'number' && latencySeconds > 0 && (
          <div style={{ textAlign: 'right', minWidth: '100px' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Latency
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {latencySeconds.toFixed(2)}s
            </div>
          </div>
        )}
      </div>

      {metrics && (
        <div style={{ marginTop: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <BarChart3 size={16} color="var(--accent-primary)" />
            <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
              Backend TrustEngine Measurements
            </h4>
          </div>

          <div className="trust-metrics-grid" data-testid="trust-metrics-grid">
            <div className="metric-card" title="Composite multi-signal trust confidence score">
              <div className="metric-name">Confidence</div>
              <div
                className="metric-value"
                style={{ color: isSupported ? 'var(--trust-supported)' : 'var(--trust-refusal)' }}
                data-testid="confidence-score"
              >
                {formatPercentage(metrics.confidence_score)}
              </div>
            </div>

            <div className="metric-card" title="Groundedness: Cross-sentence alignment with source evidence">
              <div className="metric-name">Groundedness</div>
              <div className="metric-value" data-testid="groundedness-score">
                {formatPercentage(metrics.groundedness_score)}
              </div>
            </div>

            <div className="metric-card" title="Neural relevance score from cross-encoder reranking">
              <div className="metric-name">Relevance</div>
              <div className="metric-value" data-testid="relevance-score">
                {formatPercentage(metrics.relevance_score)}
              </div>
            </div>

            <div className="metric-card" title="Query term lexical coverage ratio across top retrieved chunks">
              <div className="metric-name">Coverage</div>
              <div className="metric-value" data-testid="coverage-score">
                {formatPercentage(metrics.coverage_score)}
              </div>
            </div>

            <div className="metric-card" title="Provenance integrity across all supporting chunks">
              <div className="metric-name">Provenance</div>
              <div
                className="metric-value"
                style={{
                  fontSize: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  color: metrics.provenance_valid ? 'var(--trust-supported)' : 'var(--trust-error)',
                }}
                data-testid="provenance-valid"
              >
                {metrics.provenance_valid ? (
                  <>
                    <CheckCircle2 size={16} /> Valid
                  </>
                ) : (
                  <>
                    <XCircle size={16} /> Invalid
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!isSupported && (
        <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
          <HelpCircle size={15} style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            TrustRAG strictly enforces deterministic trust gating. When retrieved passages lack sufficient factual coverage or confidence, generation is gated to prevent hallucinations.
          </span>
        </div>
      )}
    </div>
  );
};
