import React, { useState, useRef } from 'react'; 

function MultimodalDigitRecognition() {
  const [textDigit, setTextDigit] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null); 

  const fileInputRef = useRef(null);


  const handleFileChange = (event) => {
    const file = event.target.files[0];
    console.log("handleFileChange: File selected (event.target.files[0]):", file);

    if (file) {
      const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (validExtensions.includes(fileExtension)) {
        setSelectedFile(file);
        setError(null);
        setResults(null);
        console.log("handleFileChange: Valid file. selectedFile set.");
      } else {
        setSelectedFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
          console.log("handleFileChange: Invalid file format. Input file value reset.");
        }
        setError('Choose one of these audio file formats (.wav, .mp3, .flac, .ogg).');
      }
    } else {
      console.log("handleFileChange: No file selected or input cleared.");
      setSelectedFile(null);
      setResults(null);
      setError(null);
    }
    console.log("handleFileChange: Current state after update - selectedFile:", file);
  };


  const handleSubmit = async () => {
    console.log("handleSubmit: Submitting query.");
    if (!textDigit.trim()) {
      setError('Please enter a digit (0-9).');
      return;
    }
    if (!selectedFile) {
      setError('Please choose an audio file.');
      return;
    }

    const digit = parseInt(textDigit);
    if (isNaN(digit) || digit < 0 || digit > 9) {
      setError('Please enter a valid single digit between 0 and 9.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('text_digit', textDigit);
    
    if (selectedFile) {
      formData.append('audio_file', selectedFile, selectedFile.name);
      console.log("handleSubmit: Sending selected file:", selectedFile.name);

      for (let pair of formData.entries()) {
          console.log("FormData entry:", pair[0], pair[1]); 
      }
    } else {
        console.error("handleSubmit: No audio to send!");
        setLoading(false);
        setError("No audio file available to send.");
        return;
    }

    try {
      const response = await fetch('http://localhost:8000/multimodal-digit-recognition/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error during multimodal recognition.');
      }

      const data = await response.json();
      setResults(data);
      console.log('Multimodal Recognition Results:', data);

    } catch (err) {
      console.error('Error during multimodal recognition:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="component-box">
      <h2 className="component-box-title">
        Multimodal Digit Recognition
      </h2>

      <div className="form-group">
        <label htmlFor="text-digit-input">
          Enter a digit (0-9):
        </label>
        <input
          type="text"
          id="text-digit-input"
          value={textDigit}
          onChange={(e) => setTextDigit(e.target.value)}
          placeholder="e.g.: 5"
          maxLength="1"
          pattern="[0-9]"
        />
      </div>

      <div className="form-group">
        <label htmlFor="audio-digit-upload">
          Choose an audio file (.wav, .mp3, .flac, .ogg):
        </label>
        <input
          type="file"
          id="audio-digit-upload"
          accept=".wav, .mp3, .flac, .ogg, audio/wav, audio/mpeg, audio/flac, audio/ogg"
          onChange={handleFileChange}
          ref={fileInputRef}
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
        onClick={handleSubmit}
        disabled={loading || !textDigit.trim() || !selectedFile}
        className="action-button semantic-query-button"
      >
        {loading ? (
          <span className="loading-text">
            <svg className="loading-spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Recognizing...
          </span>
        ) : (
          'Recognize Digit'
        )}
      </button>

      {results && (
        <div className="results-section">
          <h3>Multimodal Recognition Results:</h3>
          <p>Text Digit Input: <strong>{results.text_digit_input}</strong></p>
          <p>Audio Digit Recognized: <strong>{results.audio_digit_recognized_by_pinecone || 'N/A'}</strong></p>
          <p>
            Match: {' '}
            <span style={{ color: results.is_match ? 'green' : 'red', fontWeight: 'bold' }}>
              {results.is_match ? 'YES' : 'NO'}
            </span>
          </p>

          {results.recognition_details && results.recognition_details.length > 0 && (
            <div className="recognition-details-section">
              <h4>Top Audio Recognition Details:</h4>
              <ul className="results-list">
                {results.recognition_details.map((detail, index) => (
                  <li key={index} className="result-item">
                    <p>Speaker ID: <span className="speaker-id">{detail.speaker_id}</span></p>
                    <p>Score: <span className="score">{detail.score.toFixed(4)}</span></p>
                    <p>Recognized Digit from Audio: <strong>{detail.audio_digit_recognized || 'N/A'}</strong></p>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {!results.recognition_details || results.recognition_details.length === 0 && (
            <p className="info-text">No audio recognition details available.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default MultimodalDigitRecognition;
