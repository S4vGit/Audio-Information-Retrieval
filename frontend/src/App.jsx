// frontend/src/App.jsx
import React, { useState } from 'react';
import './index.css'; // Importa il file CSS di Tailwind

function App() {
  const [selectedFile, setSelectedFile] = useState(null); // Stato per il file audio selezionato
  const [results, setResults] = useState([]); // Stato per i risultati del riconoscimento
  const [loading, setLoading] = useState(false); // Stato per l'indicatore di caricamento
  const [error, setError] = useState(null); // Stato per i messaggi di errore

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Puoi fare un controllo più robusto sul tipo MIME o sull'estensione
      const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (validExtensions.includes(fileExtension)) {
        setSelectedFile(file);
        setError(null); // Pulisci errori precedenti
      } else {
        setSelectedFile(null);
        setError('Si prega di selezionare un file audio valido (.wav, .mp3, .flac, .ogg).');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Seleziona un file audio prima di caricare.');
      return;
    }

    setLoading(true); // Imposta lo stato di caricamento
    setError(null); // Pulisci errori precedenti
    setResults([]); // Pulisci risultati precedenti

    const formData = new FormData();
    formData.append('audio_file', selectedFile); // 'audio_file' deve corrispondere al nome del parametro in FastAPI

    try {
      // Assicurati che questo URL corrisponda all'indirizzo del tuo backend FastAPI
      const response = await fetch('http://localhost:8000/speaker-recognition/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Se la risposta non è OK (es. status 4xx o 5xx)
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Errore durante il caricamento dell\'audio.');
      }

      const data = await response.json();
      setResults(data.results); // Aggiorna lo stato con i risultati
      console.log('Risultati dal backend:', data.results);

    } catch (err) {
      console.error('Errore durante il caricamento:', err);
      setError(err.message); // Visualizza il messaggio di errore all'utente
    } finally {
      setLoading(false); // Disattiva lo stato di caricamento
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-10 rounded-lg p-4 bg-white bg-opacity-90 shadow-xl border border-blue-200">
        🎤 Riconoscimento Vocale Speaker 🎤
      </h1>

      <div className="bg-white p-8 md:p-10 rounded-2xl shadow-2xl w-full max-w-md border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-800 mb-7 text-center">
          Carica il tuo Audio per l'Analisi
        </h2>

        <div className="mb-7">
          <label htmlFor="speaker-recognition" className="block text-gray-700 text-sm font-semibold mb-3">
            Seleziona un file WAV (solo .wav):
          </label>
          <input
            type="file"
            id="speaker-recognition"
            accept=".wav, .mp3, .flac, .ogg, audio/*"
            onChange={handleFileChange}
            className="w-full px-4 py-2 text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
          />
          {selectedFile && (
            <p className="mt-3 text-sm text-gray-600">File selezionato: <span className="font-medium">{selectedFile.name}</span></p>
          )}
        </div>

        {error && (
          <p className="text-red-700 text-sm mb-5 p-3 bg-red-100 rounded-lg border border-red-200 shadow-sm">
            🚨 {error}
          </p>
        )}

        <button
          onClick={handleUpload}
          disabled={!selectedFile || loading}
          className={`w-full py-3 rounded-lg text-lg font-bold transition duration-300 transform hover:scale-105
            ${selectedFile && !loading
              ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg'
              : 'bg-gray-300 text-gray-600 cursor-not-allowed opacity-75'
            }`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Elaborazione in corso...
            </span>
          ) : (
            'Carica e Riconosci Speaker'
          )}
        </button>

        {results.length > 0 && (
          <div className="mt-8 pt-6 border-t border-gray-200">
            <h3 className="text-xl font-bold text-gray-800 mb-5 text-center">Risultati: Speaker Riconosciuti</h3>
            <ul className="space-y-4">
              {results.map((result, index) => (
                <li
                  key={index}
                  className="flex flex-col sm:flex-row justify-between items-center bg-blue-50 p-4 rounded-xl border border-blue-200 shadow-sm transition-transform transform hover:scale-[1.02]"
                >
                  <span className="font-semibold text-gray-900 mb-1 sm:mb-0">Speaker ID: <span className="text-blue-700">{result.speaker_id}</span></span>
                  <span className="text-gray-700 text-sm">Punteggio: <span className="font-medium">{result.score.toFixed(4)}</span></span>
                </li>
              ))}
            </ul>
            <p className="mt-6 text-center text-gray-600 text-sm">
                Il punteggio indica la somiglianza. Un punteggio più alto significa maggiore somiglianza.
            </p>
          </div>
        )}
      </div>
      <footer className="mt-10 text-gray-600 text-sm">
        Sviluppato con React (Vite) e FastAPI
      </footer>
    </div>
  );
}

export default App;