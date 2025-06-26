import React from 'react';
import './index.css'; 

import SpeakerRecognition from './components/SpeakerRecognition';
import SemanticQuery from './components/TextualQuery';

function App() {
  return (
    <div className="app-container">
      <h1 className="main-title">
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