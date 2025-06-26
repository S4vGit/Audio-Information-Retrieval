import React, { useState } from 'react';

function TextualQueryNeo4j() {
  const [textQuery, setTextQuery] = useState('');
  const [results, setResults] = useState(null); // To store both cypher and neo4j results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTextQuery = async () => {
    if (!textQuery.trim()) {
      setError('Insert a query.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null); // Clear previous results

    try {
      const response = await fetch(`http://localhost:8000/text-query/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nl_query: textQuery }), // Send as JSON body
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error during textual query.');
      }

      const data = await response.json();
      setResults(data); // Store the entire response (cypher, results)
      console.log('Result textual query:', data);

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
          Make a natural language query on speaker's metadata:
        </label>
        <input
          type="text"
          id="text-query"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          placeholder="e.g.: speakers from Italy"
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
            Querying...
          </span>
        ) : (
          'Submit Query'
        )}
      </button>

      {results && (
        <div className="results-section">
          <h3>Neo4j Query Results:</h3>
          {results.cypher && (
            <div className="cypher-query-section">
              <h4>Generated Cypher Query:</h4>
              <pre className="cypher-code"><code>{results.cypher}</code></pre>
            </div>
          )}
          {results.results && results.results.length > 0 ? (
            <ul className="results-list">
              {results.results.map((record, index) => (
                <li key={index} className="result-item">
                  {Object.entries(record).map(([key, value]) => (
                    <p key={key}>
                      <span className="result-key">{key}:</span>{' '}
                      <span className="result-value">{JSON.stringify(value)}</span>
                    </p>
                  ))}
                </li>
              ))}
            </ul>
          ) : (
            <p className="info-text">No results found for your query.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default TextualQueryNeo4j;