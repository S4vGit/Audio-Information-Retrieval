// frontend/src/App.jsx
import React from 'react';
import './index.css'; // Importa il file CSS personalizzato

// Importa i componenti (questo è lo stesso di prima, le modifiche sono interne ai componenti)
import SpeakerRecognition from './components/SpeakerRecognition';
import SemanticQuery from './components/SemanticQuery';

function App() {
  return (
    <div className="app-container"> {/* Nuova classe CSS per il contenitore principale */}
      <h1 className="main-title"> {/* Nuova classe CSS per il titolo */}
        🎤 Audio Information Retrieval System 🎤
      </h1>

      <div className="components-container">
        <SpeakerRecognition />
        <SemanticQuery />
      </div>

      <footer className="footer">
        Developed with React (Vite) and FastAPI
      </footer>
    </div>
  );
}

export default App;