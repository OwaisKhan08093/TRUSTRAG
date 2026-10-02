import { useState } from 'react';
import { Header } from './components/Header';
import type { HealthResponse } from './types/trustrag';


export function App() {
  const [health] = useState<HealthResponse | null>({
    status: 'healthy',
    version: '1.0.0',
    device: 'cpu',
    pipeline_ready: true,
  });
  const [isOnline] = useState<boolean>(true);

  return (
    <div className="app-container">
      <Header health={health} isBackendOnline={isOnline} />
      <main className="main-grid">
        <section className="glass-card">
          <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
            Welcome to TrustRAG
          </h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            Confidence-Aware Grounded RAG with Multi-Agent Verification and Deterministic Trust Gating.
          </p>
        </section>
      </main>
      <footer className="footer">
        TrustRAG System &bull; Retrieval &bull; Verification &bull; Trust Gating &bull; Grounded Generation &bull; Provenance Citations
      </footer>
    </div>
  );
}

export default App;
