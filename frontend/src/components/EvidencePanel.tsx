import React, { useState } from 'react';
import { Layers, FileCheck, Cpu, ChevronDown, ChevronRight, Hash, Clock } from 'lucide-react';
import type { CitationItem, TraceEventItem } from '../types/trustrag';

interface EvidencePanelProps {
  citations: CitationItem[];
  traceEvents?: TraceEventItem[];
  metadata?: Record<string, unknown>;
  activeCitationIndex?: number | null;
  onSelectCitation?: (citation: CitationItem) => void;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  citations,
  traceEvents = [],
  metadata = {},
  activeCitationIndex,
  onSelectCitation,
}) => {
  const [activeTab, setActiveTab] = useState<'evidence' | 'trace' | 'metadata'>('evidence');
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});

  const toggleChunk = (chunkId: string) => {
    setExpandedChunks((prev) => ({
      ...prev,
      [chunkId]: !prev[chunkId],
    }));
  };

  return (
    <div className="glass-card" data-testid="evidence-panel">
      {/* Tabs Header */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', marginBottom: '1.25rem', gap: '0.5rem' }}>
        <button
          type="button"
          className="btn"
          onClick={() => setActiveTab('evidence')}
          style={{
            padding: '0.5rem 0.85rem',
            fontSize: '0.85rem',
            borderRadius: 'var(--radius-sm) var(--radius-sm) 0 0',
            backgroundColor: activeTab === 'evidence' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
            color: activeTab === 'evidence' ? '#60a5fa' : 'var(--text-secondary)',
            borderBottom: activeTab === 'evidence' ? '2px solid #3b82f6' : '2px solid transparent',
          }}
          data-testid="tab-evidence"
        >
          <FileCheck size={15} />
          Evidence Chunks ({citations.length})
        </button>

        <button
          type="button"
          className="btn"
          onClick={() => setActiveTab('trace')}
          style={{
            padding: '0.5rem 0.85rem',
            fontSize: '0.85rem',
            borderRadius: 'var(--radius-sm) var(--radius-sm) 0 0',
            backgroundColor: activeTab === 'trace' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
            color: activeTab === 'trace' ? '#60a5fa' : 'var(--text-secondary)',
            borderBottom: activeTab === 'trace' ? '2px solid #3b82f6' : '2px solid transparent',
          }}
          data-testid="tab-trace"
        >
          <Cpu size={15} />
          Agent Trace ({traceEvents.length})
        </button>

        {Object.keys(metadata).length > 0 && (
          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab('metadata')}
            style={{
              padding: '0.5rem 0.85rem',
              fontSize: '0.85rem',
              borderRadius: 'var(--radius-sm) var(--radius-sm) 0 0',
              backgroundColor: activeTab === 'metadata' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
              color: activeTab === 'metadata' ? '#60a5fa' : 'var(--text-secondary)',
              borderBottom: activeTab === 'metadata' ? '2px solid #3b82f6' : '2px solid transparent',
            }}
            data-testid="tab-metadata"
          >
            <Layers size={15} />
            Metadata
          </button>
        )}
      </div>

      {/* Evidence Tab */}
      {activeTab === 'evidence' && (
        <div data-testid="evidence-list">
          {citations.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              No supporting evidence chunks for this query result.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {citations.map((cite) => {
                const isSelected = activeCitationIndex === cite.index;
                const isExpanded = expandedChunks[cite.chunk_id] ?? true;

                return (
                  <div
                    key={cite.chunk_id || cite.index}
                    style={{
                      background: isSelected ? 'rgba(59, 130, 246, 0.1)' : 'rgba(0, 0, 0, 0.25)',
                      border: `1px solid ${isSelected ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                      borderRadius: 'var(--radius-md)',
                      padding: '1rem',
                      transition: 'all 0.2s ease',
                    }}
                    data-testid={`evidence-item-${cite.index}`}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        cursor: 'pointer',
                      }}
                      onClick={() => {
                        toggleChunk(cite.chunk_id);
                        if (onSelectCitation) onSelectCitation(cite);
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                        <span
                          style={{
                            background: 'var(--citation-pill-bg)',
                            color: '#c4b5fd',
                            padding: '0.15rem 0.5rem',
                            borderRadius: '4px',
                            fontWeight: 700,
                            fontSize: '0.8rem',
                          }}
                        >
                          [{cite.index}]
                        </span>
                        <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                          {cite.document_name}
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        <span>
                          Page {cite.page_start === cite.page_end ? cite.page_start : `${cite.page_start}-${cite.page_end}`}
                        </span>
                        <span
                          style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            padding: '0.1rem 0.4rem',
                            borderRadius: '4px',
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                          }}
                        >
                          <Hash size={11} style={{ display: 'inline' }} />
                          {cite.chunk_id.slice(-8)}
                        </span>
                      </div>
                    </div>

                    {isExpanded && (
                      <div
                        style={{
                          marginTop: '0.75rem',
                          paddingTop: '0.75rem',
                          borderTop: '1px solid var(--border-subtle)',
                          fontSize: '0.88rem',
                          color: '#cbd5e1',
                          lineHeight: '1.55',
                          whiteSpace: 'pre-wrap',
                          fontStyle: 'italic',
                        }}
                      >
                        &ldquo;{cite.text_snippet}&rdquo;
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Trace Tab */}
      {activeTab === 'trace' && (
        <div className="trace-list" data-testid="trace-list">
          {traceEvents.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              No execution trace events available.
            </div>
          ) : (
            traceEvents.map((evt, i) => {
              const isCompleted = evt.status === 'COMPLETED';
              const isSkipped = evt.status === 'SKIPPED';
              const isFailed = evt.status === 'FAILED';

              return (
                <div key={i} className="trace-item" data-testid={`trace-item-${i}`}>
                  <div className="trace-agent">
                    <span
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: isCompleted
                          ? 'var(--trust-supported)'
                          : isSkipped
                          ? 'var(--trust-refusal)'
                          : isFailed
                          ? 'var(--trust-error)'
                          : 'var(--accent-primary)',
                      }}
                    />
                    <span>{evt.agent_name}</span>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        padding: '0.1rem 0.35rem',
                        borderRadius: '4px',
                        backgroundColor: isCompleted
                          ? 'var(--trust-supported-bg)'
                          : isSkipped
                          ? 'var(--trust-refusal-bg)'
                          : isFailed
                          ? 'var(--trust-error-bg)'
                          : 'rgba(59, 130, 246, 0.15)',
                        color: isCompleted
                          ? '#34d399'
                          : isSkipped
                          ? '#fbbf24'
                          : isFailed
                          ? '#f87171'
                          : '#60a5fa',
                        fontWeight: 600,
                      }}
                    >
                      {evt.status}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {evt.reason && (
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        {evt.reason}
                      </span>
                    )}
                    <span className="trace-latency">
                      <Clock size={12} style={{ display: 'inline', marginRight: '3px' }} />
                      {evt.latency_seconds.toFixed(3)}s
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Metadata Tab */}
      {activeTab === 'metadata' && (
        <div data-testid="metadata-view">
          <pre
            style={{
              background: 'rgba(0, 0, 0, 0.4)',
              padding: '1rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.8rem',
              fontFamily: 'monospace',
              color: '#93c5fd',
              overflowX: 'auto',
            }}
          >
            {JSON.stringify(metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
