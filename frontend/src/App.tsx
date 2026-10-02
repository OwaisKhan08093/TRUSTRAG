import { useState } from 'react';
import { Header } from './components/Header';
import { QueryInput } from './components/QueryInput';
import type { HealthResponse } from './types/trustrag';

export function App() {
  const [health] = useState<HealthResponse | null>({
    status: 'healthy',
    version: '1.0.0',
    device: 'cpu',
    pipeline_ready: true,
  });
  const [isOnline] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [submittedQuery, setSubmittedQuery] = useState<string>('');

  const handleQuerySubmit = (query: string) => {
    setSubmittedQuery(query);
    setIsLoading(true);
    // Placeholder response simulation for interface testing
    setTimeout(() => {
      setIsLoading(false);
    }, 800);
  };

  const handleReset = () => {
    setSubmittedQuery('');
    setIsLoading(false);
  };

  return (
    <div className="app-container">
      <Header health={health} isBackendOnline={isOnline} />
      <main className="main-grid">
        <QueryInput
          onSubmit={handleQuerySubmit}
          onReset={handleReset}
          isLoading={isLoading}
        />
        {submittedQuery && (
          <section className="glass-card">
            <h3 style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>Current Query:</h3>
            <p style={{ marginTop: '0.25rem', color: 'var(--text-primary)', fontWeight: 500 }}>
              {submittedQuery}
            </p>
          </section>
        )}
      </main>
      <footer className="footer">
        TrustRAG System &bull; Retrieval &bull; Verification &bull; Trust Gating &bull; Grounded Generation &bull; Provenance Citations
      </footer>
    </div>
  );
}

export default App;

