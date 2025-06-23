# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import os
import re
import librosa
import soundfile as sf
import tempfile 
import sys
import pysolr
sys.path.append(str(Path(__file__).resolve().parent)) 

from backend.pinecone.feature_extraction import extract_combined_features_vector
from backend.pinecone.pinecone_setup import initialize_pinecone_index_features
from backend.text_retrieval import load_metadata, build_textual_profiles, compute_embeddings

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

"""# Variabili globali
metadata = None # Potresti non aver più bisogno di caricare tutto in memoria se usi Solr come fonte primaria
SOLR_URL = 'http://localhost:8983/solr/audio_metadata_core'
solr = None # Inizializzeremo la connessione Solr all'avvio"""

# Inizializza Pinecone e lo scaler globalmente una volta all'avvio dell'applicazione
speaker_recognition_index = None
speaker_recognition_scaler = None

text_encoder: SentenceTransformer = None # Modello per embedding testuali
all_speaker_ids: List[str] = [] # ID degli speaker caricati
all_metadata_embeddings: np.ndarray = None # Embedding dei profili testuali
all_raw_metadata: Dict[str, Any] = {} # Metadati originali (utile per i risultati)

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


"""@app.on_event("startup")
async def load_data():
    global metadata # Potresti ancora voler caricare i metadati in memoria per altri scopi o debugging
    global solr
    try:
        # Carica i metadati se ancora necessari per altri scopi nel tuo backend
        # metadata_path = "dataset/audioMNIST_meta.txt"
        # with open(metadata_path, 'r', encoding='utf-8') as f:
        #     metadata = json.load(f)
        # print("Metadati caricati con successo (in memoria, se necessario).")

        # Inizializza la connessione a Solr
        solr = pysolr.Solr(SOLR_URL, timeout=10)
        # Testa la connessione facendo una query semplice
        solr.ping()
        print(f"Connessione a Solr stabilita su {SOLR_URL}.")

    except pysolr.SolrError as e:
        print(f"Errore di connessione a Solr: {e}. Assicurati che Solr sia in esecuzione e il core esista.")
        solr = None # Imposta a None se la connessione fallisce
    except Exception as e:
        print(f"Errore inatteso durante l'avvio: {e}")
        solr = None
"""

@app.on_event("startup")
async def startup_event():
    global solr, text_encoder, all_speaker_ids, all_metadata_embeddings, all_raw_metadata

    # 1. Inizializzazione Solr
    #print(f"Tentativo di connessione a Solr all'URL: {SOLR_URL}")
    #solr = pysolr.Solr(SOLR_URL, always_commit=False) # Commit manuale o tramite script di indicizzazione
    #try:
    #    # Tenta una query semplice per verificare la connessione
    #    solr.search("*:*", rows=0) 
    #    print("Connessione a Solr riuscita.")
    #except pysolr.SolrError as e:
    #    print(f"Errore di connessione a Solr: {e}")
        # Potresti voler fermare l'applicazione o loggare un errore critico

    # 2. Inizializzazione SentenceTransformer e caricamento embedding
    print("Inizializzazione SentenceTransformer e caricamento embedding metadati...")
    try:
        # Carica i metadati usando la funzione da ir_utils.py
        # Il percorso deve essere corretto dal punto di vista del main.py
        # main.py è in 'backend/', audioMNIST_meta.txt è in 'dataset/' (root del progetto)
        # Quindi '..' per andare su a 'project_root', poi 'dataset/audioMNIST_meta.txt'
        metadata_file_path = Path(__file__).parent.parent / "dataset/audioMNIST_meta.txt"
        
        all_raw_metadata = load_metadata(str(metadata_file_path))
        profiles = build_textual_profiles(all_raw_metadata)
        
        text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        all_speaker_ids, all_metadata_embeddings = compute_embeddings(profiles, text_encoder)
        
        print(f"Caricati e processati {len(all_speaker_ids)} profili per la ricerca semantica.")
    except Exception as e:
        print(f"Errore durante l'inizializzazione SentenceTransformer/embedding: {e}")
        # Gestisci l'errore, l'applicazione potrebbe non funzionare correttamente per le query semantiche


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

@app.get("/query-semantic/")
async def query_semantic(q: str = Query(..., description="Query di testo per la ricerca semantica")):
    """
    Esegue una ricerca semantica sui metadati degli speaker usando gli embedding.
    """
    if text_encoder is None or all_metadata_embeddings is None:
        raise HTTPException(status_code=503, detail="Modello di embedding non caricato o dati non disponibili.")

    try:
        # 1. Calcola l'embedding della query dell'utente
        query_embedding = text_encoder.encode(q, convert_to_numpy=True)

        # 2. Calcola la similarità coseno con tutti gli embedding dei metadati
        similarities = all_metadata_embeddings @ query_embedding / (
            np.linalg.norm(all_metadata_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )

        # 3. Ordina e prendi i top N risultati
        top_k = 10 # Puoi regolare quanti risultati vuoi
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            speaker_id = all_speaker_ids[idx]
            similarity_score = float(similarities[idx]) # Converti in float standard per JSON
            
            # Recupera i metadati originali per il speaker
            speaker_metadata = all_raw_metadata.get(speaker_id, {})

            results.append({
                "speaker_id": speaker_id,
                "similarity_score": round(similarity_score, 4), # Arrotonda per una migliore leggibilità
                "metadata": speaker_metadata
            })
        
        return {
            "query": q,
            "result_type": "semantic",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la ricerca semantica: {e}")

"""@app.get("/query-metadata-general-solr/")
async def query_metadata_general_solr(q: str = Query(..., min_length=1, max_length=150)):
    if solr is None:
        raise HTTPException(status_code=500, detail="Connessione a Solr non stabilita.")

    query_lower = q.lower().strip()
    
    # --- 1. Riconoscimento dell'Intento ---
    intent = None
    if any(keyword in query_lower for keyword in ["quanti", "numero di", "conta"]):
        intent = "count"
    elif any(keyword in query_lower for keyword in ["età media", "media di età"]):
        intent = "average_age"
    elif any(keyword in query_lower for keyword in ["find", "trova", "mostra", "speaker di", "speaker con"]):
        intent = "list"

    if not intent:
        intent = "list"
    
    # --- 2. Estrazione dei Filtri Strutturali e Traduzione a Solr Query Language ---
    solr_query_parts = []
    
    # Gender
    if "female" in query_lower or "women" in query_lower:
        solr_query_parts.append("gender:female")
    elif "male" in query_lower or "men" in query_lower:
        solr_query_parts.append("gender:male")

    # Age (più robusto per Solr)
    age_match = re.search(r'(tra\s+(\d+)\s+e\s+(\d+)\s+anni)|(più di\s*(\d+)\s*anni)|(meno di\s*(\d+)\s*anni)|((\d+)\s*anni)', query_lower)
    if age_match:
        if age_match.group(2) and age_match.group(3): # tra X e Y anni
            solr_query_parts.append(f"age:[{age_match.group(2)} TO {age_match.group(3)}]")
        elif age_match.group(5): # più di X anni
            solr_query_parts.append(f"age:[{int(age_match.group(5)) + 1} TO *]")
        elif age_match.group(7): # meno di X anni
            solr_query_parts.append(f"age:[* TO {int(age_match.group(7)) - 1}]")
        elif age_match.group(9): # X anni (esattamente)
            solr_query_parts.append(f"age:{age_match.group(9)}")

    # Accent
    accent_keywords = ["tedesco", "Italian", "francese", "spagnolo", "german", "english", "french", "spanish"]
    for acc in accent_keywords:
        if acc in query_lower:
            solr_query_parts.append(f"accent:{acc.capitalize()}") # Assumiamo accent capitalizzato
            break

    # Origin
    origin_keywords = ["germania", "europa", "berlino", "Italy", "münster", "dresden", "hamburg", "bremen"]
    for org in origin_keywords:
        if org in query_lower:
            solr_query_parts.append(f"origin:{org.capitalize()}") # Assumiamo origin capitalizzato
            break
    
    # Native speaker (usiamo il nome campo dello schema Solr)
    if "madrelingua si" in query_lower or "native speaker yes" in query_lower:
        solr_query_parts.append("native_speaker:yes")
    elif "madrelingua no" in query_lower or "native speaker no" in query_lower:
        solr_query_parts.append("native_speaker:no")

    # Costruisci la query principale per Solr
    if solr_query_parts:
        solr_q = " AND ".join(solr_query_parts)
    else:
        solr_q = "*:*" # Ricerca tutti i documenti se nessun filtro strutturale è stato trovato

    # --- 3. Esegui la Query su Solr ---
    try:
        # Per 'count' e 'average_age' potremmo voler solo i dati aggregati, non tutti i documenti
        if intent in ["count", "average_age"]:
            # Solr può fare aggregaizoni direttamente: stats component per media, somma, ecc.
            # Per il conteggio, basta il numero di risultati
            solr_results = solr.search(solr_q, rows=0) # rows=0 per non restituire documenti, solo metadati
            num_found = solr_results.hits
            
            if intent == "average_age":
                # Richiede una query Solr più complessa con stats
                # Solr example: q=*:*&stats=true&stats.field=age
                stats_results = solr.search(solr_q, stats=['age'], rows=0)
                age_stats = stats_results.stats.get('age', {})
                if 'mean' in age_stats:
                    average_age = age_stats['mean']
                    return {
                        "query": q,
                        "interpreted_intent": intent,
                        "solr_query": solr_q,
                        "result_message": f"L'età media degli speaker che corrispondono ai criteri è di {average_age:.2f} anni.",
                        "result_count": num_found,
                        "applied_filters": solr_query_parts
                    }
                else:
                    return {
                        "query": q,
                        "interpreted_intent": intent,
                        "solr_query": solr_q,
                        "result_message": "Impossibile calcolare l'età media per gli speaker trovati (dati insufficienti o non validi).",
                        "result_count": num_found,
                        "applied_filters": solr_query_parts
                    }
            else: # intent == "count"
                return {
                    "query": q,
                    "interpreted_intent": intent,
                    "solr_query": solr_q,
                    "result_message": f"Ci sono {num_found} speaker che corrispondono ai criteri.",
                    "result_count": num_found,
                    "applied_filters": solr_query_parts
                }

        else: # intent == "list" o default
            # Per l'elenco, recupera i documenti
            solr_results = solr.search(solr_q, rows=100) # Limita a 100 risultati per l'elenco
            
            speaker_details = []
            for doc in solr_results.docs:
                speaker_details.append(doc) # Solr restituisce i documenti come dizionari

            speaker_ids_list = [doc.get('id') for doc in solr_results.docs]

            response_message = ""
            if speaker_ids_list:
                response_message = f"Gli speaker che corrispondono ai criteri sono: {', '.join(speaker_ids_list)}."
            else:
                response_message = "Nessuno speaker trovato con i criteri specificati."

            return {
                "query": q,
                "interpreted_intent": intent,
                "solr_query": solr_q,
                "result_message": response_message,
                "result_count": solr_results.hits,
                "applied_filters": solr_query_parts,
                "speaker_details": speaker_details
            }

    except pysolr.SolrError as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la query Solr: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore inatteso durante la query: {e}")

"""

# Puoi aggiungere altri endpoint se necessario
@app.get("/")
async def read_root():
    return {"message": "API voicerecognition!"}