import React, { useState } from 'react';

function SemanticQuery() {
  const [textQuery, setTextQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTextQuery = async () => {
    if (!textQuery.trim()) {
      setError('Insert a query.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await fetch(`http://localhost:8000/query-semantic/?q=${encodeURIComponent(textQuery)}`, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error during textual query.');
      }

      const data = await response.json();
      setResults(data.results);
      console.log('Result textual query:', data.results);

    } catch (err) {
      console.error('Error textual query:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="component-box">
      <h2 className="component-box-title">
        Textual Query on Speaker Metadata
      </h2>

      <div className="form-group">
        <label htmlFor="text-query">
          Make a query on speaker's metadata:
        </label>
        <input
          type="text"
          id="text-query"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          placeholder="e.g.: old people with english accent"
        />
      </div>

      {error && (
        <p className="error-message">
          🚨 {error}
        </p>
      )}

      <button
        onClick={handleTextQuery}
        disabled={loading}
        className="action-button semantic-query-button"
      >
        {loading ? (
          <span className="loading-text">
            <svg className="loading-spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Ricerca in corso...
          </span>
        ) : (
          'Submit Query'
        )}
      </button>

      {results.length > 0 && (
        <div className="results-section">
          <h3>Text Query Results:</h3>
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <p>Speaker ID: <span className="speaker-id">{result.speaker_id}</span></p>
                <p>Similarity Score: <span className="similarity-score">{result.similarity_score.toFixed(4)}</span></p>
                {result.metadata && (
                  <div className="metadata-details">
                    <p>Gender: {result.metadata.gender || 'N/A'}</p>
                    <p>Age: {result.metadata.age || 'N/A'}</p>
                    <p>Accent: {result.metadata.accent || 'N/A'}</p>
                    <p>Origin: {result.metadata.origin || 'N/A'}</p>
                  </div>
                )}
              </li>
            ))}
          </ul>
          <p className="info-text">
            Il punteggio di similarità indica quanto il profilo dello speaker è vicino alla tua domanda.
          </p>
        </div>
      )}
    </div>
  );
}

export default SemanticQuery;