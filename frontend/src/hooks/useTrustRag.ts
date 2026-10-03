import { useState, useCallback, useEffect } from 'react';
import { apiClient, ApiError, NetworkError } from '../services/api';
import type {
  QueryResponse,
  HealthResponse,
  QueryStatus,
  QueryRequest,
  DocumentInfo,
  DocumentUploadResponse,
} from '../types/trustrag';

export interface UseTrustRagReturn {
  status: QueryStatus;
  response: QueryResponse | null;
  health: HealthResponse | null;
  isBackendOnline: boolean;
  errorMessage: string | null;
  currentStepMessage: string;
  documents: DocumentInfo[];
  isUploading: boolean;
  uploadSuccessMessage: string | null;
  uploadErrorMessage: string | null;
  submitQuery: (query: string, options?: Partial<QueryRequest>) => Promise<void>;
  uploadDocument: (file: File) => Promise<DocumentUploadResponse | null>;
  loadDocuments: () => Promise<void>;
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

  // Document upload state
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState<string | null>(null);
  const [uploadErrorMessage, setUploadErrorMessage] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const data = await apiClient.getHealth();
      setHealth(data);
      setIsBackendOnline(true);
    } catch {
      setIsBackendOnline(false);
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    try {
      const res = await apiClient.listDocuments();
      setDocuments(res.documents || []);
    } catch {
      // Non-blocking if documents endpoint fails
    }
  }, []);

  useEffect(() => {
    checkHealth();
    loadDocuments();
    const interval = setInterval(checkHealth, 30000); // Poll health every 30s
    return () => clearInterval(interval);
  }, [checkHealth, loadDocuments]);

  const uploadDocument = useCallback(
    async (file: File): Promise<DocumentUploadResponse | null> => {
      setIsUploading(true);
      setUploadSuccessMessage(null);
      setUploadErrorMessage(null);

      try {
        const result = await apiClient.uploadDocument(file);
        setUploadSuccessMessage(`✓ Document "${result.filename}" processed successfully (${result.chunks_created} chunks created). Ready for questions!`);
        await loadDocuments();
        return result;
      } catch (err: unknown) {
        if (err instanceof ApiError || err instanceof Error) {
          setUploadErrorMessage(err.message || 'Failed to upload and process document.');
        } else {
          setUploadErrorMessage('An unexpected error occurred during document upload.');
        }
        return null;
      } finally {
        setIsUploading(false);
      }
    },
    [loadDocuments]
  );

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
    setUploadSuccessMessage(null);
    setUploadErrorMessage(null);
  }, []);

  return {
    status,
    response,
    health,
    isBackendOnline,
    errorMessage,
    currentStepMessage,
    documents,
    isUploading,
    uploadSuccessMessage,
    uploadErrorMessage,
    submitQuery,
    uploadDocument,
    loadDocuments,
    reset,
    checkHealth,
  };
}

