import React, { useRef, useState } from 'react';
import type { DocumentInfo } from '../types/trustrag';

interface DocumentUploadProps {
  documents: DocumentInfo[];
  isUploading: boolean;
  uploadSuccessMessage: string | null;
  uploadErrorMessage: string | null;
  onUpload: (file: File) => Promise<unknown>;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  documents,
  isUploading,
  uploadSuccessMessage,
  uploadErrorMessage,
  onUpload,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        await onUpload(file);
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await onUpload(file);
      // Reset input value so same file can be re-uploaded if needed
      e.target.value = '';
    }
  };

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }} data-testid="document-upload-section">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>📄</span> Ingest Documents
        </h2>
        {documents.length > 0 && (
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            {documents.length} document{documents.length > 1 ? 's' : ''} indexed
          </span>
        )}
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        data-testid="upload-dropzone"
        style={{
          border: isDragOver ? '2px dashed var(--primary-color)' : '2px dashed var(--border-color)',
          backgroundColor: isDragOver ? 'rgba(59, 130, 246, 0.08)' : 'var(--bg-card)',
          borderRadius: '0.75rem',
          padding: '1.5rem',
          textAlign: 'center',
          cursor: isUploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          opacity: isUploading ? 0.7 : 1,
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={isUploading}
          data-testid="file-input"
        />

        {isUploading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div className="spinner" style={{ width: '28px', height: '28px' }} />
            <p style={{ fontWeight: 500, color: 'var(--primary-color)' }}>
              Processing document: Parsing PDF, chunking & generating neural embeddings...
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ fontSize: '2rem' }}>📥</div>
            <p style={{ fontWeight: 500, fontSize: '1rem', color: 'var(--text-main)' }}>
              Drag & drop PDF here, or <span style={{ color: 'var(--primary-color)', textDecoration: 'underline' }}>browse</span>
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Supports PDF documents up to 25MB with automatic chunking & indexing
            </p>
          </div>
        )}
      </div>

      {/* Upload Messages */}
      {uploadSuccessMessage && (
        <div
          data-testid="upload-success-message"
          style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            borderRadius: '0.5rem',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            color: '#4ade80',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          {uploadSuccessMessage}
        </div>
      )}

      {uploadErrorMessage && (
        <div
          data-testid="upload-error-message"
          style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            borderRadius: '0.5rem',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#f87171',
            fontSize: '0.9rem',
          }}
        >
          ⚠️ {uploadErrorMessage}
        </div>
      )}

      {/* Ingested Documents List */}
      {documents.length > 0 && (
        <div style={{ marginTop: '1.25rem' }}>
          <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Currently Indexed Documents
          </h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {documents.map((doc) => (
              <div
                key={doc.document_id}
                data-testid={`doc-chip-${doc.document_id}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.4rem 0.75rem',
                  borderRadius: '0.5rem',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-color)',
                  fontSize: '0.85rem',
                }}
              >
                <span>📑</span>
                <span style={{ fontWeight: 500 }}>{doc.document_name}</span>
                <span
                  style={{
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    color: '#60a5fa',
                    padding: '0.1rem 0.4rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                  }}
                >
                  {doc.chunks_count} chunks
                </span>
                <span
                  style={{
                    backgroundColor: 'rgba(34, 197, 94, 0.15)',
                    color: '#4ade80',
                    padding: '0.1rem 0.4rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                  }}
                >
                  ✓ Ready
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
