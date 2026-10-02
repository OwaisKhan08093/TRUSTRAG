# TrustRAG React Frontend

A modern, responsive React + TypeScript interface for the **TrustRAG** Confidence-Aware Grounded RAG system with Deterministic Trust Gating.

---

## Features

* **Grounded Query Interface**: Natural language input with quick-select example queries, keyboard shortcuts (`Enter`), reset capability, and real-time state feedback.
* **Deterministic Trust Decision UI**: Direct visualization of backend `TrustEngine` measurements (`SUPPORTED` vs `INSUFFICIENT_EVIDENCE`), composite confidence, groundedness, cross-encoder relevance, lexical coverage, and provenance integrity.
* **Verifiable Citation & Provenance**: Interactive citation tags (`[1]`, `[2]`) linked directly to source document chunk cards with page ranges and text snippets.
* **Multi-Agent Pipeline Visualization**: Visual telemetry mapping the 5-agent execution workflow (`RETRIEVE` &rarr; `EVIDENCE` &rarr; `TRUST` &rarr; `GENERATE` &rarr; `CITE`) reflecting actual agent trace latencies and gating status.
* **Clean State Handling**: Dedicated UX states for `IDLE`, `LOADING` (with multi-agent stage progress), `SUCCESS`, `INSUFFICIENT_EVIDENCE`, `API_ERROR`, and `NETWORK_ERROR`.

---

## Tech Stack

* **Core**: React 19 + TypeScript
* **Build Tool**: Vite 8
* **Styling**: Vanilla CSS Design System with dark-mode glassmorphism and HSL tailored tokens
* **Icons**: Lucide React
* **Testing**: Vitest + React Testing Library + jsdom

---

## Directory Structure

```text
frontend/
├── src/
│   ├── components/
│   │   ├── Header.tsx              # Header with API connection status
│   │   ├── QueryInput.tsx          # Query input, quick prompts & buttons
│   │   ├── StatusMessage.tsx       # Loading pulse and user-friendly error banners
│   │   ├── PipelineVisualization.tsx# 5-stage multi-agent pipeline flow
│   │   ├── TrustDecision.tsx       # TrustEngine metrics and gating banner
│   │   ├── AnswerView.tsx          # Grounded prose with clickable citation tags
│   │   └── EvidencePanel.tsx       # Chunk provenance, agent traces & metadata
│   ├── hooks/
│   │   └── useTrustRag.ts          # Custom React hook for API state & polling
│   ├── pages/
│   │   └── HomePage.tsx            # Main layout and view composition
│   ├── services/
│   │   └── api.ts                  # Dedicated FastAPI client service
│   ├── test/
│   │   ├── setup.ts                # Vitest environment setup
│   │   ├── api.test.ts             # API client unit tests
│   │   ├── components.test.tsx     # Component rendering & behavior tests
│   │   └── integration.test.tsx    # End-to-end frontend integration tests
│   ├── types/
│   │   └── trustrag.ts             # Pydantic-aligned TypeScript interfaces
│   ├── App.tsx                     # Top-level application component
│   ├── index.css                   # Core design tokens and styles
│   └── main.tsx                    # Application entry point
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## Environment Variables

Configure `.env` or `.env.local` in `frontend/`:

```env
# URL of the TrustRAG FastAPI backend server
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Getting Started

### 1. Install Dependencies

```bash
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

The application will be accessible at `http://localhost:5173`.

### 3. Build for Production

```bash
npm run build
```

### 4. Run Test Suite

```bash
npm test
```
