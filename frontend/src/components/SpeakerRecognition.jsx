import React, { useState } from 'react';

function SpeakerRecognition() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (validExtensions.includes(fileExtension)) {
        setSelectedFile(file);
        setError(null);
      } else {
        setSelectedFile(null);
        setError('Choose one of this audio file format (.wav, .mp3, .flac, .ogg).');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Choose an audio file to upload.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    const formData = new FormData();
    formData.append('audio_file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/speaker-recognition/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error during audio upload.');
      }

      const data = await response.json();
      setResults(data.results);
      console.log('Backend results:', data.results);

    } catch (err) {
      console.error('Error during loading:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="component-box">
      <h2 className="component-box-title">
        Speaker Recognition
      </h2>

      <div className="form-group">
        <label htmlFor="audio-upload">
          Choose a file audio (.wav, .mp3, .flac, .ogg):
        </label>
        <input
          type="file"
          id="audio-upload"
          accept=".wav, .mp3, .flac, .ogg, audio/wav, audio/mpeg, audio/flac, audio/ogg"
          onChange={handleFileChange}
        />
        {selectedFile && (
          <p className="selected-file-info">File selected: <span>{selectedFile.name}</span></p>
        )}
      </div>

      {error && (
        <p className="error-message">
          🚨 {error}
        </p>
      )}

      <button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
        className="action-button speaker-recognition-button"
      >
        {loading ? (
          <span className="loading-text">
            <svg className="loading-spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
          </span>
        ) : (
          'Recognize Speaker'
        )}
      </button>

      {results.length > 0 && (
        <div className="results-section">
          <h3>Speaker Recognition Results</h3>
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <p>Speaker ID: <span className="speaker-id">{result.speaker_id}</span></p>
                <p>Score: <span className="score">{result.score.toFixed(4)}</span></p>
              </li>
            ))}
          </ul>
          <p className="info-text">
            The score indicates the confidence level of the speaker recognition. A higher score means a more confident match.
          </p>
        </div>
      )}
    </div>
  );
}

export default SpeakerRecognition;