import React, { useState } from 'react';

function TextAudioQuery() {
  const [nlQueryText, setNlQueryText] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [audioFileName, setAudioFileName] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioFileName(file.name);
    } else {
      setAudioFile(null);
      setAudioFileName('');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setResults(null);

    if (!nlQueryText.trim()) {
      setError("Per favore, inserisci una query testuale.");
      return;
    }
    if (!audioFile) {
      setError("Per favore, carica un file audio.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('nl_query_text', nlQueryText);
    formData.append('audio_file', audioFile);

    try {
      const response = await fetch('http://localhost:8000/multimodal-query/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Errore durante la richiesta API multimodale.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
      console.error('Errore nella richiesta multimodale:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="component-box">
      <h2>Query Multimodale</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="nlQueryText">Query Testuale (es: "donne con accento tedesco"):</label>
          <input
            type="text"
            id="nlQueryText"
            value={nlQueryText}
            onChange={(e) => setNlQueryText(e.target.value)}
            placeholder="Inserisci la query testuale"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="audioFile">Carica Audio:</label>
          <input
            type="file"
            id="audioFile"
            accept=".wav, .mp3, .flac, .ogg, .webm"
            onChange={handleFileChange}
            disabled={loading}
          />
          {audioFileName && (
            <p className="selected-file-info">File selezionato: {audioFileName}</p>
          )}
        </div>

        {error && <p className="error-message">{error}</p>}

        <button
          type="submit"
          className="action-button multimodal-query-button"
          disabled={loading || !nlQueryText.trim() || !audioFile}
        >
          {loading ? (
            <>
              <span className="loading-spinner"></span> Elaborazione...
            </>
          ) : (
            'Esegui Query Multimodale'
          )}
        </button>
      </form>

      {results && (
        <div className="results-section">
          <h3>Risultati Multimodali</h3>
          <p><strong>ID Speaker Riconosciuto dall'Audio:</strong> {results.audio_recognized_speaker_id || 'N/A'}</p>
          <p><strong>ID Speaker dalla Query Testuale:</strong> {results.text_query_speaker_ids.length > 0 ? results.text_query_speaker_ids.join(', ') : 'Nessuno'}</p>
          <p><strong>Stato del Match:</strong> {results.match_status}</p>
        </div>
      )}
    </div>
  );
}

export default TextAudioQuery;
