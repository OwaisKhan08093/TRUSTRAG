import React from 'react';
import { ShieldCheck, Activity } from 'lucide-react';
import type { HealthResponse } from '../types/trustrag';


interface HeaderProps {
  health: HealthResponse | null;
  isBackendOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, isBackendOnline }) => {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="brand-icon">
          <ShieldCheck size={26} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h1 className="brand-title">TrustRAG</h1>
            <span className="brand-badge">Deterministic Grounding</span>
          </div>
          <p className="brand-subtitle">
            Multi-Agent Grounded Retrieval-Augmented Generation with Trust Gating
          </p>
        </div>
      </div>

      <div className="header-status" title={isBackendOnline ? 'Connected to FastAPI Backend' : 'FastAPI Backend Disconnected'}>
        <div className={`status-dot ${isBackendOnline ? '' : 'offline'}`} />
        <span style={{ fontWeight: 500 }}>
          {isBackendOnline ? (
            <>
              API Online <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>v{health?.version || '1.0.0'}</span>
            </>
          ) : (
            <span style={{ color: 'var(--trust-error)' }}>API Offline</span>
          )}
        </span>
        {health?.device && (
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Activity size={12} /> {health.device}
          </span>
        )}
      </div>
    </header>
  );
};
