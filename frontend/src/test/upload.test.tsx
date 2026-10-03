import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentUpload } from '../components/DocumentUpload';
import type { DocumentInfo } from '../types/trustrag';

describe('DocumentUpload Component', () => {
  const mockDocuments: DocumentInfo[] = [
    {
      document_id: 'doc_dpdp_01',
      document_name: 'DPDP_Act_2023.pdf',
      chunks_count: 8,
      pages_count: 25,
      total_words: 4200,
      status: 'ready',
    },
  ];

  it('1. renders upload UI with drag and drop zone and title', () => {
    render(
      <DocumentUpload
        documents={[]}
        isUploading={false}
        uploadSuccessMessage={null}
        uploadErrorMessage={null}
        onUpload={vi.fn()}
      />
    );

    expect(screen.getByTestId('document-upload-section')).toBeInTheDocument();
    expect(screen.getByTestId('upload-dropzone')).toBeInTheDocument();
    expect(screen.getByText(/Drag & drop PDF here/i)).toBeInTheDocument();
  });

  it('2. displays processing state when isUploading is true', () => {
    render(
      <DocumentUpload
        documents={[]}
        isUploading={true}
        uploadSuccessMessage={null}
        uploadErrorMessage={null}
        onUpload={vi.fn()}
      />
    );

    expect(screen.getByText(/Processing document: Parsing PDF, chunking/i)).toBeInTheDocument();
  });

  it('3. displays upload success message when processing succeeds', () => {
    render(
      <DocumentUpload
        documents={mockDocuments}
        isUploading={false}
        uploadSuccessMessage="✓ Document 'machine_learning.pdf' processed successfully (42 chunks created). Ready for questions!"
        uploadErrorMessage={null}
        onUpload={vi.fn()}
      />
    );

    expect(screen.getByTestId('upload-success-message')).toBeInTheDocument();
    expect(screen.getByText(/processed successfully/i)).toBeInTheDocument();
  });

  it('4. displays upload error message when upload fails', () => {
    render(
      <DocumentUpload
        documents={[]}
        isUploading={false}
        uploadSuccessMessage={null}
        uploadErrorMessage="Unsupported file format 'script.exe'. Only PDF documents (.pdf) are currently supported."
        onUpload={vi.fn()}
      />
    );

    expect(screen.getByTestId('upload-error-message')).toBeInTheDocument();
    expect(screen.getByText(/Unsupported file format/i)).toBeInTheDocument();
  });

  it('5. displays document-ready status and chunk count in catalog list', () => {
    render(
      <DocumentUpload
        documents={mockDocuments}
        isUploading={false}
        uploadSuccessMessage={null}
        uploadErrorMessage={null}
        onUpload={vi.fn()}
      />
    );

    expect(screen.getByText('DPDP_Act_2023.pdf')).toBeInTheDocument();
    expect(screen.getByText('8 chunks')).toBeInTheDocument();
    expect(screen.getByText('✓ Ready')).toBeInTheDocument();
    expect(screen.getByText('1 document indexed')).toBeInTheDocument();
  });

  it('6. triggers onUpload when file is selected via input', async () => {
    const onUploadMock = vi.fn().mockResolvedValue({});
    render(
      <DocumentUpload
        documents={[]}
        isUploading={false}
        uploadSuccessMessage={null}
        uploadErrorMessage={null}
        onUpload={onUploadMock}
      />
    );

    const input = screen.getByTestId('file-input');
    const file = new File(['%PDF-1.4 test data'], 'sample.pdf', { type: 'application/pdf' });

    fireEvent.change(input, { target: { files: [file] } });
    expect(onUploadMock).toHaveBeenCalledWith(file);
  });
});
