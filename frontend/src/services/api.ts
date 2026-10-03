/**
 * Dedicated API Client Service for TrustRAG FastAPI Backend.
 */

import type {
  QueryRequest,
  QueryResponse,
  HealthResponse,
  DocumentUploadResponse,
  DocumentListResponse,
} from '../types/trustrag';

export class ApiError extends Error {
  public status?: number;
  public details?: unknown;

  constructor(message: string, status?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class TrustRagApiClient {
  private baseUrl: string;
  private defaultTimeout: number;

  constructor(
    baseUrl: string = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
    defaultTimeout: number = 60000 // 60s for local inference / pipeline
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.defaultTimeout = defaultTimeout;
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeoutMs: number = this.defaultTimeout
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          ...(options.headers || {}),
        },
      });

      clearTimeout(timer);

      let payload: unknown;
      const text = await response.text();
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch {
          throw new ApiError(
            `Malformed JSON received from backend (${response.status})`,
            response.status,
            text
          );
        }
      }

      if (!response.ok) {
        const errorDetail =
          payload && typeof payload === 'object' && 'detail' in payload
            ? String((payload as { detail: unknown }).detail)
            : `HTTP request failed with status ${response.status}`;
        throw new ApiError(errorDetail, response.status, payload);
      }

      return payload as T;
    } catch (err: unknown) {
      clearTimeout(timer);
      if (err instanceof ApiError) {
        throw err;
      }
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          throw new NetworkError(`Request timed out after ${timeoutMs / 1000}s`);
        }
        throw new NetworkError(err.message || 'Unable to connect to TrustRAG backend');
      }
      throw new NetworkError('Unknown network failure');
    }
  }

  /**
   * Health check endpoint: GET /health
   */
  public async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health', {
      method: 'GET',
    }, 5000);
  }

  /**
   * Query execution endpoint: POST /query
   */
  public async submitQuery(requestData: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  /**
   * Document upload endpoint: POST /documents (multipart/form-data)
   */
  public async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/documents`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000); // 120s for ingestion/embedding

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          Accept: 'application/json',
          // Note: browser automatically sets multipart/form-data boundary
        },
      });

      clearTimeout(timer);
      const text = await response.text();
      let payload: unknown;
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch {
          throw new ApiError(`Malformed response from upload: ${text}`, response.status);
        }
      }

      if (!response.ok) {
        const errorDetail =
          payload && typeof payload === 'object' && 'detail' in payload
            ? String((payload as { detail: unknown }).detail)
            : `Document upload failed with status ${response.status}`;
        throw new ApiError(errorDetail, response.status, payload);
      }

      return payload as DocumentUploadResponse;
    } catch (err: unknown) {
      clearTimeout(timer);
      if (err instanceof ApiError) throw err;
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          throw new NetworkError('Document upload & ingestion timed out');
        }
        throw new NetworkError(err.message || 'Unable to upload document');
      }
      throw new NetworkError('Unknown upload error');
    }
  }

  /**
   * Document catalog endpoint: GET /documents
   */
  public async listDocuments(): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>('/documents', {
      method: 'GET',
    }, 10000);
  }
}

// Export singleton instance for standard use
export const apiClient = new TrustRagApiClient();

