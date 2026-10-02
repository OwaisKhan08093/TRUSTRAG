import { useState, useCallback, useEffect } from 'react';
import { apiClient, ApiError, NetworkError } from '../services/api';
import type {
  QueryResponse,
  HealthResponse,
  QueryStatus,
  QueryRequest,
} from '../types/trustrag';

export interface UseTrustRagReturn {
  status: QueryStatus;
  response: QueryResponse | null;
  health: HealthResponse | null;
  isBackendOnline: boolean;
  errorMessage: string | null;
  currentStepMessage: string;
  submitQuery: (query: string, options?: Partial<QueryRequest>) => Promise<void>;
  reset: () => void;
  checkHealth: () => Promise<void>;
}

export function useTrustRag(): UseTrustRagReturn {
  const [status, setStatus] = useState<QueryStatus>('IDLE');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentStepMessage, setCurrentStepMessage] = useState<string>('');

  const checkHealth = useCallback(async () => {
    try {
      const data = await apiClient.getHealth();
      setHealth(data);
      setIsBackendOnline(true);
    } catch {
      setIsBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Poll health every 30s
    return () => clearInterval(interval);
  }, [checkHealth]);

  const submitQuery = useCallback(
    async (query: string, options: Partial<QueryRequest> = {}) => {
      const trimmed = query.trim();
      if (!trimmed) {
        setErrorMessage('Query cannot be empty');
        setStatus('API_ERROR');
        return;
      }

      setStatus('LOADING');
      setErrorMessage(null);
      setCurrentStepMessage('Querying TrustRAG pipeline: Retrieving & verifying evidence...');

      try {
        const result = await apiClient.submitQuery({
          query: trimmed,
          filter_cited_only: true,
          ...options,
        });

        setResponse(result);
        setIsBackendOnline(true);

        if (result.is_refusal || result.decision === 'INSUFFICIENT_EVIDENCE') {
          setStatus('INSUFFICIENT_EVIDENCE');
        } else {
          setStatus('SUCCESS');
        }
      } catch (err: unknown) {
        if (err instanceof NetworkError) {
          setStatus('NETWORK_ERROR');
          setIsBackendOnline(false);
          setErrorMessage(err.message || 'Unable to connect to the TrustRAG backend service.');
        } else if (err instanceof ApiError) {
          setStatus('API_ERROR');
          setErrorMessage(err.message || 'The TrustRAG API returned an error.');
        } else if (err instanceof Error) {
          setStatus('API_ERROR');
          setErrorMessage(err.message);
        } else {
          setStatus('API_ERROR');
          setErrorMessage('An unexpected error occurred.');
        }
      } finally {
        setCurrentStepMessage('');
      }
    },
    []
  );

  const reset = useCallback(() => {
    setStatus('IDLE');
    setResponse(null);
    setErrorMessage(null);
    setCurrentStepMessage('');
  }, []);

  return {
    status,
    response,
    health,
    isBackendOnline,
    errorMessage,
    currentStepMessage,
    submitQuery,
    reset,
    checkHealth,
  };
}
