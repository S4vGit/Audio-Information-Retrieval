import React, { useState } from 'react'; // Import useState
import './index.css'; 

import SpeakerRecognition from './components/SpeakerRecognition';
import SemanticQuery from './components/TextualQuery';
import MultimodalDigitRecognition from './components/MultimodalQuery';
import TextAudioQuery from './components/TextAudioQuery'; 

function App() {
  const [textualQueryHasResults, setTextualQueryHasResults] = useState(false);

  return (
    <div className="app-container">
      <h1 className="main-title">
        🎤 Audio Information Retrieval System 🎤
      </h1>

      <div className="components-container">
        <SpeakerRecognition textualQueryHasResults={textualQueryHasResults} />
        <SemanticQuery onResultsChange={setTextualQueryHasResults} /> 
        <MultimodalDigitRecognition />
        <TextAudioQuery />
      </div>

      <footer className="footer">
        Developed with React (Vite) and FastAPI
      </footer>
    </div>
  );
}

export default App;