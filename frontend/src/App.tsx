import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { useTrustRag } from './hooks/useTrustRag';

export function App() {
  const trustRag = useTrustRag();

  return (
    <div className="app-container">
      <Header
        health={trustRag.health}
        isBackendOnline={trustRag.isBackendOnline}
      />
      <main>
        <HomePage trustRag={trustRag} />
      </main>
      <footer className="footer">
        TrustRAG System &bull; Retrieval &bull; Verification &bull; Trust Gating &bull; Grounded Generation &bull; Provenance Citations
      </footer>
    </div>
  );
}

export default App;
