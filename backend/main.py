# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import os
import librosa
import soundfile as sf
import tempfile 
import sys
sys.path.append(str(Path(__file__).resolve().parent)) 

from backend.pinecone.feature_extraction import extract_combined_features_vector
from backend.pinecone.pinecone_setup import initialize_pinecone_index_features

# --- Configurazione delle Feature e dell'Indice (DEVE CORRISPONDERE ALLA TUA FASE DI INDICIzzAZIONE) ---
# Queste feature e n_mfcc devono essere le stesse usate per creare l'indice Pinecone
# e per addestrare lo scaler.
USED_FEATURES = ['mfcc', 'sc', 'rms', 'zcr'] # Esempio: ['mfcc', 'sc', 'rms', 'zcr'] o solo ['mfcc']
N_MFCC_USED = 13               # Il numero di MFCC usati durante l'indicizzazione
SPEAKER_RECOGNITION_INDEX_NAME = "speaker-recognition-mfcc-sc-rms-zcr" # Il nome del tuo indice Pinecone

# --- Mappatura delle dimensioni per calcolare la dimensione totale del vettore ---
FEATURE_DIMENSIONS = {
    'mfcc': N_MFCC_USED, # Usa N_MFCC_USED per la dimensione di MFCC
    'sc': 1,
    'rms': 1,
    'zcr': 1,
}

def calculate_vector_dimension(features_list: list, n_mfcc: int) -> int:
    """Calcola la dimensione totale del vettore in base alle feature."""
    total_dimension = 0
    for feature in features_list:
        if feature == 'mfcc':
            total_dimension += n_mfcc
        elif feature in FEATURE_DIMENSIONS:
            total_dimension += FEATURE_DIMENSIONS[feature]
        else:
            raise ValueError(f"Feature '{feature}' non riconosciuta o dimensione non definita.")
    return total_dimension

VECTOR_DIM = calculate_vector_dimension(USED_FEATURES, N_MFCC_USED)

# Inizializza Pinecone e lo scaler globalmente una volta all'avvio dell'applicazione
speaker_recognition_index = None
speaker_recognition_scaler = None

# --- Funzione per la conversione audio ---
def convert_audio_to_wav(input_audio_path: Path, output_wav_path: Path):
    """
    Convert a given audio file to WAV format using librosa and soundfile.

    Args:
        input_audio_path (Path): Path to the input audio file.
        output_wav_path (Path): Path to the output WAV file.

    Returns:
        Path: Path to the converted WAV file, or None if conversion fails.
    """
    if not output_wav_path.lower().endswith('.wav'):
        print(f"Erorr: output_wav_path must end with '.wav', but '{output_wav_path}'")
        return None

    try:
        # Carica il file audio. librosa può gestire vari formati.
        # sr=None mantiene la frequenza di campionamento originale.
        y, sr = librosa.load(str(input_audio_path), sr=None)

        # Scrivi i dati audio in un file WAV
        sf.write(str(output_wav_path), y, sr)

        print(f"Successfully converted '{input_audio_path.name}' in '{output_wav_path.name}'")
        return output_wav_path
    except Exception as e:
        print(f"Error during convertion of audio file '{input_audio_path.name}': {e}")
        return None

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.on_event("startup")
async def startup_event():
    """
    Funzione eseguita all'avvio dell'applicazione FastAPI.
    Carica lo scaler e inizializza l'indice Pinecone.
    """
    global speaker_recognition_index, speaker_recognition_scaler

    # Carica lo scaler
    # Assicurati che questo percorso sia corretto rispetto alla radice del progetto
    scalers_dir = Path("backend/scalers")
    feature_names_str = "_".join(USED_FEATURES)
    
    # Controlla il nome esatto del file dello scaler.
    # Se lo hai salvato come 'scaler_feature_mfcc_sc_little.pkl', usa questo:
    scaler_filename = scalers_dir / f"scaler_feature_{feature_names_str}.pkl"
    # OPPURE, se hai usato "_little.pkl" nel nome del file dello scaler:
    # scaler_filename = scalers_dir / f"scaler_feature_{feature_names_str}_little.pkl"


    if not scaler_filename.exists():
        raise RuntimeError(f"File scaler non trovato: {scaler_filename}. Assicurati che esista e che il percorso sia corretto.")
    
    try:
        speaker_recognition_scaler = joblib.load(scaler_filename)
        print(f"Scaler caricato con successo da: {scaler_filename}")
    except Exception as e:
        raise RuntimeError(f"Errore durante il caricamento dello scaler da {scaler_filename}: {e}")

    # Inizializza l'indice Pinecone
    try:
        speaker_recognition_index = initialize_pinecone_index_features(SPEAKER_RECOGNITION_INDEX_NAME, VECTOR_DIM)
        print(f"Indice Pinecone '{SPEAKER_RECOGNITION_INDEX_NAME}' inizializzato (Dimensione: {VECTOR_DIM}).")
    except Exception as e:
        raise RuntimeError(f"Errore durante l'inizializzazione dell'indice Pinecone: {e}")

# Modello Pydantic per la risposta dell'API
class SpeakerResult(BaseModel):
    speaker_id: str
    score: float

@app.post("/speaker-recognition/")
async def upload_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint per caricare un file audio, estrarre feature, scalarle,
    e interrogare l'indice Pinecone per il riconoscimento dello speaker.
    """
    if not speaker_recognition_scaler or not speaker_recognition_index:
        raise HTTPException(status_code=500, detail="Backend non completamente inizializzato. Riprova tra un momento.")

    original_file_extension = Path(audio_file.filename).suffix.lower()
    
    # Lista delle estensioni audio supportate
    supported_audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']

    if original_file_extension not in supported_audio_extensions:
        raise HTTPException(status_code=400, detail=f"Formato file non supportato. Sono supportati solo: {', '.join(supported_audio_extensions)}.")

    tmp_original_file_path = None
    tmp_wav_file_path = None
    audio_for_processing_path = None
    
    """if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Sono supportati solo i file WAV.")"""

    """# Crea un file temporaneo per salvare l'audio caricato
    # Questo è più sicuro e gestisce meglio i file di grandi dimensioni
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(await audio_file.read())
        tmp_file_path = Path(tmp_file.name)"""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp_original_file:
            tmp_original_file.write(await audio_file.read())
            tmp_original_file_path = Path(tmp_original_file.name)
        
        # Se il file non è già WAV, convertilo
        if original_file_extension != '.wav':
            tmp_wav_file_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            converted_path = convert_audio_to_wav(tmp_original_file_path, tmp_wav_file_path)
            
            if converted_path is None:
                raise HTTPException(status_code=500, detail="Errore durante la conversione del file audio in WAV.")
            audio_for_processing_path = converted_path
        else:
            audio_for_processing_path = tmp_original_file_path # Usa direttamente il file WAV caricato
            
        # 1. Estrai le feature dall'audio
        # Assicurati che n_mfcc corrisponda a quello usato in fase di training/indicizzazione
        raw_features = extract_combined_features_vector(audio_for_processing_path, USED_FEATURES, N_MFCC_USED)
        if raw_features is None:
            raise HTTPException(status_code=500, detail="Errore nell'estrazione delle feature dall'audio.")

        # 2. Scala le feature usando lo scaler caricato
        # raw_features.reshape(1, -1) è necessario perché scaler.transform si aspetta un input 2D
        scaled_features = speaker_recognition_scaler.transform(raw_features.reshape(1, -1))[0]

        # 3. Interroga Pinecone con le feature scalate
        query_results = speaker_recognition_index.query(
            vector=scaled_features.tolist(), # Converti il vettore NumPy in una lista Python
            top_k=5, # Ottieni i primi 5 speaker più simili
            include_metadata=True # Includi i metadati per ottenere lo speaker_id
        )

        results = []
        # Estrai i risultati
        for match in query_results.matches:
            speaker_id = match.metadata.get('speaker_id', 'Unknown')
            score = match.score
            results.append(SpeakerResult(speaker_id=speaker_id, score=score))
        
        # Ordina i risultati per punteggio in ordine decrescente
        results.sort(key=lambda x: x.score, reverse=True)

        return {"message": "Audio elaborato con successo", "results": results}

    except Exception as e:
        print(f"Errore durante l'elaborazione dell'audio: {e}")
        # Restituisci un errore HTTP 500 in caso di eccezioni non gestite
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {e}")
    finally:
        # Pulisci i file temporanei
        if tmp_original_file_path and tmp_original_file_path.exists():
            os.unlink(tmp_original_file_path)
        if tmp_wav_file_path and tmp_wav_file_path.exists():
            os.unlink(tmp_wav_file_path)

# Puoi aggiungere altri endpoint se necessario
@app.get("/")
async def read_root():
    return {"message": "API voicerecognition!"}