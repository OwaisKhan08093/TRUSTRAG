import React, { useState } from 'react';
import { Copy, Check, ShieldAlert, Sparkles, BookOpen } from 'lucide-react';
import type { QueryResponse, CitationItem } from '../types/trustrag';


interface AnswerViewProps {
  response: QueryResponse;
  onCitationClick?: (citation: CitationItem) => void;
  selectedCitationIndex?: number | null;
}

export const AnswerView: React.FC<AnswerViewProps> = ({
  response,
  onCitationClick,
  selectedCitationIndex,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  /**
   * Parses markdown/text to highlight citation markers like [1], [2], etc.
   */
  const renderFormattedAnswer = (answerText: string) => {
    // Regular expression matching citation markers like [1], [1, 2], [1][2]
    const regex = /(\[(\d+(?:,\s*\d+)*)\])/g;
    const parts: (string | React.ReactNode)[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(answerText)) !== null) {
      if (match.index > lastIndex) {
        parts.push(answerText.substring(lastIndex, match.index));
      }

      const rawCitation = match[1];
      const indicesStr = match[2];
      const indices = indicesStr.split(',').map((s) => parseInt(s.trim(), 10));

      parts.push(
        <span
          key={`cite-${match.index}`}
          className="citation-tag"
          onClick={() => {
            if (onCitationClick && response.citations.length > 0) {
              const matched = response.citations.find((c) => indices.includes(c.index));
              if (matched) onCitationClick(matched);
            }
          }}
          title={`View supporting evidence for citation ${rawCitation}`}
        >
          {rawCitation}
        </span>
      );

      lastIndex = match.index + rawCitation.length;
    }

    if (lastIndex < answerText.length) {
      parts.push(answerText.substring(lastIndex));
    }

    return parts;
  };

  const isRefusal = response.is_refusal || response.decision === 'INSUFFICIENT_EVIDENCE';

  return (
    <div className="glass-card answer-box" data-testid="answer-view">
      <div className="answer-header">
        <div className="answer-title">
          {isRefusal ? (
            <>
              <ShieldAlert size={20} color="#fbbf24" />
              <span>TrustRAG Verification Notice</span>
            </>
          ) : (
            <>
              <Sparkles size={20} color="#60a5fa" />
              <span>TrustRAG Grounded Answer</span>
            </>
          )}
        </div>

        <button
          className="btn btn-secondary"
          onClick={handleCopy}
          style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
          title="Copy answer text"
          data-testid="copy-answer-btn"
        >
          {copied ? (
            <>
              <Check size={14} color="#10b981" /> Copied
            </>
          ) : (
            <>
              <Copy size={14} /> Copy Answer
            </>
          )}
        </button>
      </div>

      {isRefusal ? (
        <div className="refusal-notice" data-testid="refusal-notice">
          <h4>
            <ShieldAlert size={18} /> Insufficient Grounding Evidence
          </h4>
          <p className="answer-body">{response.answer}</p>
        </div>
      ) : (
        <div className="answer-body" data-testid="answer-text">
          {renderFormattedAnswer(response.answer)}
        </div>
      )}

      {/* Citations block if available */}
      {response.citations && response.citations.length > 0 && (
        <div className="citations-container" data-testid="citations-container">
          <div className="citations-title">
            <BookOpen size={16} color="#8b5cf6" />
            <span>Verified Source Citations ({response.citations.length})</span>
          </div>

          <div className="citations-list">
            {response.citations.map((citation) => {
              const isSelected = selectedCitationIndex === citation.index;
              return (
                <div
                  key={citation.index}
                  id={`citation-${citation.index}`}
                  className="citation-card"
                  style={{
                    borderColor: isSelected ? 'var(--citation-pill)' : undefined,
                    backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.12)' : undefined,
                  }}
                  data-testid={`citation-card-${citation.index}`}
                >
                  <div className="citation-card-header">
                    <span className="citation-ref">
                      [{citation.index}] {citation.document_name}
                    </span>
                    <span className="citation-page">
                      Page {citation.page_start === citation.page_end ? citation.page_start : `${citation.page_start}-${citation.page_end}`}
                    </span>
                  </div>
                  {citation.text_snippet && (
                    <div className="citation-snippet">
                      &ldquo;{citation.text_snippet}&rdquo;
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
