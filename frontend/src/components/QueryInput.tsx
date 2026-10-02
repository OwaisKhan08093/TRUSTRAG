import React, { useState } from 'react';
import { Send, RotateCcw, Sparkles } from 'lucide-react';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  onReset: () => void;
  isLoading: boolean;
  disabled?: boolean;
}

const EXAMPLE_QUERIES = [
  'What is personal data under the DPDP Act 2023?',
  'What are the primary obligations of a Data Fiduciary?',
  'Who is the current President of Mars?', // Refusal case
];

export const QueryInput: React.FC<QueryInputProps> = ({
  onSubmit,
  onReset,
  isLoading,
  disabled = false,
}) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isLoading || disabled) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleReset = () => {
    setQuery('');
    onReset();
  };

  const handleSelectExample = (example: string) => {
    setQuery(example);
    if (!isLoading && !disabled) {
      onSubmit(example);
    }
  };

  const isSubmitDisabled = !query.trim() || isLoading || disabled;

  return (
    <div className="glass-card query-section">
      <form onSubmit={handleSubmit} className="query-form" data-testid="query-form">
        <div className="input-wrapper">
          <textarea
            id="query-input"
            data-testid="query-input"
            className="query-textarea"
            placeholder="Ask TrustRAG a question (e.g., 'What are the rights of a Data Principal?'). Press Enter to submit..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || disabled}
            rows={3}
            aria-label="Ask TrustRAG a question"
          />
        </div>

        <div className="query-controls">
          <div className="quick-prompts">
            <span className="quick-prompt-label">
              <Sparkles size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
              Try asking:
            </span>
            {EXAMPLE_QUERIES.map((example, idx) => (
              <button
                key={idx}
                type="button"
                className="prompt-pill"
                onClick={() => handleSelectExample(example)}
                disabled={isLoading || disabled}
                title={`Ask: "${example}"`}
              >
                {example.length > 36 ? `${example.slice(0, 36)}...` : example}
              </button>
            ))}
          </div>

          <div className="action-buttons">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleReset}
              disabled={isLoading || disabled || (!query && true)}
              data-testid="query-reset-btn"
              title="Clear input and results"
            >
              <RotateCcw size={16} />
              Reset
            </button>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitDisabled}
              data-testid="query-submit-btn"
            >
              {isLoading ? (
                <>
                  <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                  Verifying & Generating...
                </>
              ) : (
                <>
                  <Send size={16} />
                  Ask TrustRAG
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
