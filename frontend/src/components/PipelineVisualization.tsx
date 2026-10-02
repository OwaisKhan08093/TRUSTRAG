import React from 'react';
import { Search, Filter, ShieldCheck, Sparkles, BookOpen, ArrowRight, CheckCircle2, Clock, Ban } from 'lucide-react';
import type { TraceEventItem, QueryStatus } from '../types/trustrag';

interface PipelineVisualizationProps {
  status: QueryStatus;
  traceEvents?: TraceEventItem[];
  decision?: string;
  isRefusal?: boolean;
}

interface StepDefinition {
  id: string;
  name: string;
  shortName: string;
  icon: React.ReactNode;
  agentPatterns: string[];
}

const PIPELINE_STEPS: StepDefinition[] = [
  {
    id: 'retrieve',
    name: 'Retrieval Agent',
    shortName: 'RETRIEVE',
    icon: <Search size={16} />,
    agentPatterns: ['retrieval', 'hybridretriever', 'retriever'],
  },
  {
    id: 'evidence',
    name: 'Evidence Agent',
    shortName: 'EVIDENCE',
    icon: <Filter size={16} />,
    agentPatterns: ['evidence', 'rerank', 'crossencoder'],
  },
  {
    id: 'trust',
    name: 'Trust Agent',
    shortName: 'TRUST',
    icon: <ShieldCheck size={16} />,
    agentPatterns: ['trust', 'trustengine', 'trustagent'],
  },
  {
    id: 'generate',
    name: 'Generation Agent',
    shortName: 'GENERATE',
    icon: <Sparkles size={16} />,
    agentPatterns: ['generation', 'generator', 'llm'],
  },
  {
    id: 'cite',
    name: 'Citation Agent',
    shortName: 'CITE',
    icon: <BookOpen size={16} />,
    agentPatterns: ['citation', 'cite', 'provenance'],
  },
];

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({
  status,
  traceEvents = [],
  decision,
  isRefusal = false,
}) => {
  const isQueryActive = status === 'LOADING';
  const hasResults = status === 'SUCCESS' || status === 'INSUFFICIENT_EVIDENCE';

  const getStepState = (step: StepDefinition) => {
    if (!hasResults && !isQueryActive) {
      return { state: 'idle', label: '', latency: null };
    }

    if (isQueryActive) {
      return { state: 'active', label: 'Processing', latency: null };
    }

    // Match against real trace events if available
    const matchedTrace = traceEvents.find((evt) =>
      step.agentPatterns.some((pattern) =>
        evt.agent_name.toLowerCase().includes(pattern)
      )
    );

    if (matchedTrace) {
      const statusUpper = matchedTrace.status.toUpperCase();
      if (statusUpper === 'COMPLETED') {
        return { state: 'completed', label: 'Completed', latency: matchedTrace.latency_seconds };
      }
      if (statusUpper === 'SKIPPED') {
        return { state: 'skipped', label: 'Gated / Skipped', latency: matchedTrace.latency_seconds };
      }
      if (statusUpper === 'FAILED') {
        return { state: 'failed', label: 'Failed', latency: matchedTrace.latency_seconds };
      }
      return { state: 'active', label: statusUpper, latency: matchedTrace.latency_seconds };
    }

    // Fallback: If refusal and it's the generate step
    if ((isRefusal || decision === 'INSUFFICIENT_EVIDENCE') && step.id === 'generate') {
      return { state: 'skipped', label: 'Trust Gated', latency: null };
    }

    if (hasResults) {
      return { state: 'completed', label: 'Executed', latency: null };
    }

    return { state: 'idle', label: '', latency: null };
  };

  return (
    <div className="glass-card" data-testid="pipeline-visualization">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>TrustRAG Multi-Agent Architecture</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            ({hasResults ? 'Verified Execution' : 'Deterministic Gating Pipeline'})
          </span>
        </h3>
        {hasResults && (
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '0.2rem 0.55rem',
              borderRadius: '9999px',
              backgroundColor: isRefusal ? 'var(--trust-refusal-bg)' : 'var(--trust-supported-bg)',
              color: isRefusal ? '#fbbf24' : '#34d399',
              border: `1px solid ${isRefusal ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
            }}
          >
            {isRefusal ? 'Gated at Trust Step' : 'All Agents Grounded'}
          </span>
        )}
      </div>

      <div className="pipeline-flow" data-testid="pipeline-flow">
        {PIPELINE_STEPS.map((step, idx) => {
          const { state, label, latency } = getStepState(step);
          const isLast = idx === PIPELINE_STEPS.length - 1;

          return (
            <React.Fragment key={step.id}>
              <div className={`pipeline-step ${state}`} data-testid={`pipeline-step-${step.id}`}>
                <div className="step-icon-wrap" title={`${step.name} - ${label || 'Standby'}`}>
                  {state === 'completed' ? (
                    <CheckCircle2 size={18} color="var(--trust-supported)" />
                  ) : state === 'skipped' ? (
                    <Ban size={18} color="var(--trust-refusal)" />
                  ) : (
                    step.icon
                  )}
                </div>

                <div className="step-name">{step.shortName}</div>

                {label && (
                  <span
                    style={{
                      fontSize: '0.65rem',
                      color:
                        state === 'completed'
                          ? 'var(--trust-supported)'
                          : state === 'skipped'
                          ? 'var(--trust-refusal)'
                          : 'var(--text-highlight)',
                      fontWeight: 600,
                    }}
                  >
                    {label}
                  </span>
                )}

                {typeof latency === 'number' && (
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                    <Clock size={10} style={{ display: 'inline', marginRight: '2px' }} />
                    {latency.toFixed(2)}s
                  </span>
                )}
              </div>

              {!isLast && (
                <div className="step-arrow">
                  <ArrowRight size={16} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
