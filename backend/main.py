from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from backend.models import SpeakerResult, NLQuery
import joblib
import numpy as np
import os
import re
import librosa
import soundfile as sf
import tempfile 
import sys
import openai
sys.path.append(str(Path(__file__).resolve().parent)) 

from backend.pinecone.feature_extraction import extract_combined_features_vector
from backend.pinecone.pinecone_setup import initialize_pinecone_index_features
from backend.utils import load_metadata, compute_metrics
from neo4j_connector import Neo4jConnector


# --- Feature and Index Configuration ---
USED_FEATURES = ['mfcc', 'sc', 'rms', 'zcr'] # Feature used for speaker recognition.
N_MFCC_USED = 13 # The number of MFCCs used during indexing
SPEAKER_RECOGNITION_INDEX_NAME = "speaker-recognition-mfcc-sc-rms-zcr" # The name of Pinecone index

# --- Mapping of dimensions to calculate the total vector dimension ---
FEATURE_DIMENSIONS = {
    'mfcc': N_MFCC_USED, # Use N_MFCC_USED for the MFCC dimension
    'sc': 1,
    'rms': 1,
    'zcr': 1,
}

# --- Load metadata ---
metadata = load_metadata()

# --- Function to calculate the vector dimension ---
def calculate_vector_dimension(features_list: list, n_mfcc: int) -> int:
    """ Compute the total dimension of the feature vector based on the features used. 
    
    Args:
        features_list (list): List of features to be used.
        n_mfcc (int): Number of MFCCs to be used if 'mfcc' is in the list.
        
    Returns:
        int: Total dimension of the feature vector.
    """
    total_dimension = 0
    for feature in features_list:
        if feature == 'mfcc':
            total_dimension += n_mfcc
        elif feature in FEATURE_DIMENSIONS:
            total_dimension += FEATURE_DIMENSIONS[feature]
        else:
            raise ValueError(f"Feature '{feature}' not recognized.")
    return total_dimension

VECTOR_DIM = calculate_vector_dimension(USED_FEATURES, N_MFCC_USED)

# --- Global Variables ---
speaker_recognition_index = None
speaker_recognition_scaler = None
neo4j_connector = None
client = None

text_encoder: SentenceTransformer = None # SentenceTransformer model for text encoding
all_speaker_ids: List[str] = [] # IDs of all speakers in the dataset
all_metadata_embeddings: np.ndarray = None # Embeddings of all metadata (useful for text queries)
all_raw_metadata: Dict[str, Any] = {} # Original metadata (useful for results)

# --- Function to convert audio files to WAV format ---
def convert_audio_to_wav(input_audio_path: Path, output_wav_path: Path):
    """
    Convert a given audio file to WAV format using librosa and soundfile.

    Args:
        input_audio_path (Path): Path to the input audio file.
        output_wav_path (Path): Path to the output WAV file.

    Returns:
        Path: Path to the converted WAV file, or None if conversion fails.
    """
    if not str(output_wav_path).lower().endswith('.wav'):
        print(f"Erorr: output_wav_path must end with '.wav', but '{output_wav_path}'")
        return None

    try:
        # Load the audio file
        y, sr = librosa.load(str(input_audio_path), sr=None)

        # Write the audio data to a WAV file
        sf.write(str(output_wav_path), y, sr)

        print(f"Successfully converted '{input_audio_path.name}' in '{output_wav_path.name}'")
        return output_wav_path
    except Exception as e:
        print(f"Error during convertion of audio file '{input_audio_path.name}': {e}")
        return None

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Startup Event for scaler, Pinecone, Neo4j and LM Studio ---
@app.on_event("startup")
async def startup_event():
    """
    Startup event to initialize the Pinecone index, scaler, Neo4j connection, and LM Studio client.
    This function is called when the FastAPI application starts.
    """
    global speaker_recognition_index, speaker_recognition_scaler, neo4j_connector, client

    # Load scaler from file
    scalers_dir = Path("backend/scalers")
    feature_names_str = "_".join(USED_FEATURES)

    # Recrete the scaler filename based on the features used
    scaler_filename = scalers_dir / f"scaler_feature_{feature_names_str}.pkl"

    if not scaler_filename.exists():
        raise RuntimeError(f"Scaler file not found: {scaler_filename}. Please ensure the scaler has been trained and saved correctly.")
    
    try:
        speaker_recognition_scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded successfully: {scaler_filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading scaler from {scaler_filename}: {e}")

    # Initialize Pinecone index
    try:
        speaker_recognition_index = initialize_pinecone_index_features(SPEAKER_RECOGNITION_INDEX_NAME, VECTOR_DIM)
        print(f"Initialized Pinecone index '{SPEAKER_RECOGNITION_INDEX_NAME}' (Dimension: {VECTOR_DIM}).")
    except Exception as e:
        raise RuntimeError(f"Error initializing Pinecone index: {e}")
    
    # Initialize Neo4j connection
    try:
        neo4j_connector = Neo4jConnector()
    except Exception as e:
        raise RuntimeError(f"Error connecting to Neo4j: {e}")
    
    # Initialize LM Studio client
    try:
        client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        print("LM Studio client initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Error connecting to LM Studio client: {e}")


# --- Endpoint for speaker recognition ---
@app.post("/speaker-recognition/")
async def upload_audio(audio_file: UploadFile = File(...)):
    """
    Recognizes the speaker from the uploaded audio file.
    
    Args:
        audio_file (UploadFile): The audio file to be processed.
    
    Returns:
        JSONResponse: A response containing the recognition results.
    """
    if not speaker_recognition_scaler or not speaker_recognition_index:
        raise HTTPException(status_code=500, detail="Pinecone index or scaler not initialized. Please check the server logs.")

    original_file_extension = Path(audio_file.filename).suffix.lower()
    
    supported_audio_extensions = ['.wav', '.mp3', '.flac', '.ogg'] # List of supported audio file extensions

    if original_file_extension not in supported_audio_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported formats are: {', '.join(supported_audio_extensions)}.")

    tmp_original_file_path = None
    tmp_wav_file_path = None
    audio_for_processing_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp_original_file:
            tmp_original_file.write(await audio_file.read())
            tmp_original_file_path = Path(tmp_original_file.name)
        
        # If the uploaded file is not a WAV file, convert it to WAV format
        if original_file_extension != '.wav':
            tmp_wav_file_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            converted_path = convert_audio_to_wav(tmp_original_file_path, tmp_wav_file_path)
            
            if converted_path is None:
                raise HTTPException(status_code=500, detail="Error converting audio file to WAV format.")
            audio_for_processing_path = converted_path
        else:
            audio_for_processing_path = tmp_original_file_path # Use the original file path if it's already in WAV format

        # 1. Extract features from the audio
        # Ensure that n_mfcc matches what was used during training/indexing
        raw_features = extract_combined_features_vector(audio_for_processing_path, USED_FEATURES, N_MFCC_USED)
        if raw_features is None:
            raise HTTPException(status_code=500, detail="Error extracting features from audio.")

        # 2. Scale the features using the loaded scaler
        scaled_features = speaker_recognition_scaler.transform(raw_features.reshape(1, -1))[0]

        # 3. Query Pinecone with the scaled features
        query_results = speaker_recognition_index.query(
            vector=scaled_features.tolist(), # Convert the NumPy vector to a Python list
            top_k=5, # Get the top 5 most similar speakers
            include_metadata=True # Include metadata 
        )
        
        # --- DEBUG ---
        print(f"Query results: {query_results}")
        metrics = compute_metrics(query_results, audio_file.filename) # Compute metrics for evaluation

        results = []
        # Extract results
        for match in query_results.matches:
            speaker_id = match.metadata.get('speaker_id', 'Unknown')
            score = match.score
            results.append(SpeakerResult(speaker_id=speaker_id, score=score))

        # Sort results by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)

        return {"message": "Audio processed successfully", "results": results, "metrics": metrics}

    except Exception as e:
        print(f"Error processing audio: {e}")
        # Return a 500 HTTP error for unhandled exceptions
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # Clean up temporary files
        if tmp_original_file_path and tmp_original_file_path.exists():
            os.unlink(tmp_original_file_path)
        if tmp_wav_file_path and tmp_wav_file_path.exists():
            os.unlink(tmp_wav_file_path)

@app.get("/query-semantic/")
async def query_semantic(q: str = Query(..., description="Query text for semantic search")):
    """
    Executes a semantic search on speaker metadata using embeddings.
    
    Args:
        q (str): The query text to search for in the speaker metadata.
        
    Returns:
        JSONResponse: A response containing the search results.
    """
    if text_encoder is None or all_metadata_embeddings is None:
        raise HTTPException(status_code=503, detail="Text encoder or metadata embeddings not initialized. Please check the server logs.")

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

# --- Endpoint for textual query ---
@app.post("/text-query/")
async def text_query(query: NLQuery):
    """ Executes a textual query against the Neo4j database using LM Studio to generate Cypher queries.
    
    Args:
        query (NLQuery): The natural language query to be converted to Cypher.
    
    Returns:
        JSONResponse: A response containing the generated Cypher query and the results from Neo4j.
    """
    
    SYSTEM_PROMPT = """
You are given the schema of a Neo4j graph database with the following entities and relationships:
- Speaker nodes with properties: id, age (integer), gender (male or female), native_speaker (yes or no)
- Accent nodes with property: name
- Location nodes with properties: city, country
- RecordingRoom nodes with property: name
Relations:
- (s:Speaker)-[:HAS_ACCENT]-(a:Accent)
- (s:Speaker)-[:RECORDED_IN {date}]-(r:RecordingRoom)
- (s:Speaker)-[:FROM_LOCATION]-(l:Location)
Generate ONLY the valid Cypher query that fulfills the user's requirement in one single statement. 
IMPORTANT: For numeric comparisons, use operators like <, >, <=, >= directly (e.g., s.age < 30), do NOT use functions like lt(), gt(), etc.
IMPORTANT: For numeric ranges, do NOT use BETWEEN ... AND ...; instead, use property >= value1 AND property <= value2 (e.g., s.age >= 25 AND s.age <= 35).
KEEP IN MIND: the operator "!=" doesn't work, use "<>" instead. 
For property comparisons other than equality, use the WHERE clause. Do not use comparison operators inside curly braces.
DO NOT add conditions not required.
Do NOT rename the entities or relationships.
Do not include any explanations or code comments."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.nl_query + "\n Answer only with the Cypher query."}
    ]

    try:
        resp = client.chat.completions.create(  # Call LLM Studio to generate Cypher
            model="local-model",
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            stream=False
            )
        cypher = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {e}")
    # Execute Cypher against Neo4j
    try:
        with neo4j_connector.driver.session() as session:
            result = session.run(cypher)
            records = [record.data() for record in result]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cypher execution error: {e}")
    return JSONResponse(content={"cypher": cypher, "results": records})









# --- Endpoint for multimodal digit recognition ---
@app.post("/multimodal-digit-recognition/")
async def multimodal_digit_recognition(
    text_digit: str = Form(...), # Ricevi la cifra testuale come Form data
    audio_file: UploadFile = File(...) # Ricevi il file audio
):
    """
    Performs multimodal digit recognition by comparing a text digit with an audio digit.
    
    Args:
        text_digit (str): The digit provided as text (0-9).
        audio_file (UploadFile): The audio file containing the spoken digit.
        
    Returns:
        JSONResponse: A response indicating if the digit matches and recognition details.
    """
    if not speaker_recognition_scaler or not speaker_recognition_index:
        raise HTTPException(status_code=500, detail="Pinecone index or scaler not initialized. Please check the server logs.")

    # Validazione della cifra testuale
    if not text_digit.isdigit() or not (0 <= int(text_digit) <= 9):
        raise HTTPException(status_code=400, detail="Text digit must be a single digit between 0 and 9.")

    original_file_extension = Path(audio_file.filename).suffix.lower()
    supported_audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.webm']

    if original_file_extension not in supported_audio_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported formats are: {', '.join(supported_audio_extensions)}.")

    tmp_original_file_path = None
    tmp_wav_file_path = None
    audio_for_processing_path = None

    try:
        # Salva il file audio temporaneamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp_original_file:
            tmp_original_file.write(await audio_file.read())
            tmp_original_file_path = Path(tmp_original_file.name)
        
        # Converte in WAV se necessario
        if original_file_extension != '.wav':
            tmp_wav_file_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            converted_path = convert_audio_to_wav(tmp_original_file_path, tmp_wav_file_path)
            
            if converted_path is None:
                raise HTTPException(status_code=500, detail="Error converting audio file to WAV format.")
            audio_for_processing_path = converted_path
        else:
            audio_for_processing_path = tmp_original_file_path

        # 1. Estrai le feature dall'audio
        raw_features = extract_combined_features_vector(audio_for_processing_path, USED_FEATURES, N_MFCC_USED)
        if raw_features is None:
            raise HTTPException(status_code=500, detail="Error extracting features from audio.")

        # 2. Scala le feature
        scaled_features = speaker_recognition_scaler.transform(raw_features.reshape(1, -1))[0]

        # 3. Interroga Pinecone con le feature scalate
        # Cerchiamo i top k speakers più simili includendo i metadati
        query_results = speaker_recognition_index.query(
            vector=scaled_features.tolist(),
            top_k=5, # Ottieni i top 5 risultati
            include_metadata=True # È cruciale includere i metadati per ottenere la cifra
        )

        matched_digit = None
        best_score = 0.0
        recognition_results = []
        
        # Estrai i risultati di Pinecone e cerca la cifra pronunciata
        for match in query_results.matches:
            speaker_id = match.metadata.get('speaker_id', 'Unknown')
            audio_digit_from_pinecone = match.metadata.get('digit') 
            score = match.score

            processed_audio_digit = None
            if audio_digit_from_pinecone is not None:
                try:
                    # Converte in intero e poi in stringa per consistenza
                    # Gestisce sia i float (es. 4.0 -> 4 -> '4') che gli interi (es. 4 -> '4')
                    processed_audio_digit = str(int(audio_digit_from_pinecone))
                except (ValueError, TypeError):
                    # Fallback nel caso in cui non sia un numero o non possa essere convertito
                    processed_audio_digit = str(audio_digit_from_pinecone)

            recognition_results.append({
                "speaker_id": speaker_id,
                "score": score,
                "audio_digit_recognized": processed_audio_digit # Usa il digit formattato
            })

            # Trova il digit con il punteggio più alto
            if score > best_score:
                best_score = score
                matched_digit = processed_audio_digit # Usa il digit formattato per il miglior match

        is_match = False
        if matched_digit and matched_digit == text_digit:
            is_match = True

        return JSONResponse(content={
            "text_digit_input": text_digit,
            "audio_digit_recognized_by_pinecone": matched_digit,
            "is_match": is_match,
            "recognition_details": recognition_results # Dettagli completi dei match di Pinecone
        })

    except Exception as e:
        print(f"Error during multimodal digit recognition: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # Pulisci i file temporanei
        if tmp_original_file_path and tmp_original_file_path.exists():
            os.unlink(tmp_original_file_path)
        if tmp_wav_file_path and tmp_wav_file_path.exists():
            os.unlink(tmp_wav_file_path)





# --- Endpoint for combined text and audio (multimodal) query ---
@app.post("/multimodal-query/")
async def multimodal_query(
    nl_query_text: str = Form(..., description="Natural language query for text retrieval"),
    audio_file: UploadFile = File(...)
):
    """
    Processes a natural language query for text retrieval and an audio file for speaker recognition,
    then verifies if the recognized speaker ID from audio is present in the results of the text query.

    Args:
        nl_query_text (str): The natural language query for the textual search.
        audio_file (UploadFile): The audio file for speaker recognition.

    Returns:
        JSONResponse: A response containing the recognized audio speaker ID,
                      speaker IDs from text query, and a match status.
    """
    # --- Part 1: Process Text Query (similar to /text-query/) ---
    if neo4j_connector is None or client is None: 
        raise HTTPException(status_code=500, detail="Neo4j connector or LLM client not initialized. Please check the server logs.")  

    SYSTEM_PROMPT = """  
You are given the schema of a Neo4j graph database with the following entities and relationships:  
- Speaker nodes with properties: id, age (integer), gender (male or female), native_speaker (yes or no)  
- Accent nodes with property: name  
- Location nodes with properties: city, country  
- RecordingRoom nodes with property: name# 
Relations:  
- (s:Speaker)-[:HAS_ACCENT]-(a:Accent)  
- (s:Speaker)-[:RECORDED_IN {date}]-(r:RecordingRoom)  
- (s:Speaker)-[:FROM_LOCATION]-(l:Location) 
Generate ONLY the valid Cypher query that fulfills the user's requirement in one single statement. 
IMPORTANT: For numeric comparisons, use operators like <, >, <=, >= directly (e.g., s.age < 30), do NOT use functions like lt(), gt(), etc. 
IMPORTANT: For numeric ranges, do NOT use BETWEEN ... AND ...; instead, use property >= value1 AND property <= value2 (e.g., s.age >= 25 AND s.age <= 35).
KEEP IN MIND: the operator "!=" doesn't work, use "<>" instead. 
For property comparisons other than equality, use the WHERE clause. Do not use comparison operators inside curly braces. 
DO NOT add conditions not required. 
Do NOT rename the entities or relationships.
Do not include any explanations or code comments.""" 

    messages = [ 
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": nl_query_text + "\n Answer only with the Cypher query."}  
    ]  

    text_query_speaker_ids = []
    try:
        resp = client.chat.completions.create( 
            model="local-model",  
            messages=messages,  
            temperature=0.3,  
            max_tokens=500, 
            stream=False 
        )  
        cypher = resp.choices[0].message.content.strip()  
        with neo4j_connector.driver.session() as session:  
            result = session.run(cypher) 

            # --- Debugging ---
            print(f"Generated Cypher Query: {cypher}")
            records_list = [record.data() for record in result] 
            print(f"Neo4j Raw Records: {records_list}")
            # --- End of debugging prints ---
            
            for record in records_list: 
                if 's' in record and 'id' in record['s']:
                    text_query_speaker_ids.append(record['s']['id'])
                elif 'speaker_id' in record: # Check for direct speaker_id return
                    text_query_speaker_ids.append(record['speaker_id'])
                elif 'id' in record: # Check for direct id return
                    text_query_speaker_ids.append(record['id'])
            text_query_speaker_ids = list(set(text_query_speaker_ids)) # Remove duplicates
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text query: {e}")

    # --- Part 2: Process Audio ---
    if not speaker_recognition_scaler or not speaker_recognition_index: # 
        raise HTTPException(status_code=500, detail="Pinecone index or scaler not initialized. Please check the server logs.") 

    original_file_extension = Path(audio_file.filename).suffix.lower() 
    supported_audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.webm']  

    if original_file_extension not in supported_audio_extensions: 
        raise HTTPException(status_code=400, detail=f"Unsupported audio file format. Supported formats are: {', '.join(supported_audio_extensions)}.")  

    tmp_original_file_path = None
    tmp_wav_file_path = None
    audio_for_processing_path = None
    recognized_audio_speaker_id = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp_original_file:  
            tmp_original_file.write(await audio_file.read())  
            tmp_original_file_path = Path(tmp_original_file.name)  

        if original_file_extension != '.wav':  
            tmp_wav_file_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)  
            converted_path = convert_audio_to_wav(tmp_original_file_path, tmp_wav_file_path)  
            if converted_path is None:  
                raise HTTPException(status_code=500, detail="Error converting audio file to WAV format.")  
            audio_for_processing_path = converted_path  
        else:  
            audio_for_processing_path = tmp_original_file_path  

        raw_features = extract_combined_features_vector(audio_for_processing_path, USED_FEATURES, N_MFCC_USED)  
        if raw_features is None:  
            raise HTTPException(status_code=500, detail="Error extracting features from audio.") 

        scaled_features = speaker_recognition_scaler.transform(raw_features.reshape(1, -1))[0] 

        query_results = speaker_recognition_index.query(  
            vector=scaled_features.tolist(),  
            top_k=1, # Get only the top 1 most similar speaker
            include_metadata=True
        )

        if query_results.matches:  
            recognized_audio_speaker_id = query_results.matches[0].metadata.get('speaker_id', 'Unknown')  

    except Exception as e:
        print(f"Error processing audio for recognition: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during audio processing: {e}")
    finally:
        if tmp_original_file_path and tmp_original_file_path.exists():
            os.unlink(tmp_original_file_path)
        if tmp_wav_file_path and tmp_wav_file_path.exists():
            os.unlink(tmp_wav_file_path)

    # --- Part 3: Verification and Final Output ---
    match_status_message = "Audio does not belong to any of the selected speaker"
    matched_id_params = None

    if recognized_audio_speaker_id and recognized_audio_speaker_id in text_query_speaker_ids:
        # Retrieve parameters for the matched ID from the original dataset (audioMNIST_meta.txt)
        matched_id_params = metadata.get(recognized_audio_speaker_id, {}) 

        if matched_id_params:
            params_str = ", ".join(f"{k}: {v}" for k, v in matched_id_params.items())
            match_status_message = f"Audio belongs to {recognized_audio_speaker_id}: {params_str}"
        else:
            match_status_message = f"Audio belongs to {recognized_audio_speaker_id}, but details could not be retrieved from metadata."

    print(f"Final text_query_speaker_ids before JSONResponse: {text_query_speaker_ids}")

    return JSONResponse(content={
        "audio_recognized_speaker_id": recognized_audio_speaker_id,
        "text_query_speaker_ids": text_query_speaker_ids,
        "match_status": match_status_message
    })


# --- Root endpoint ---
@app.get("/")
async def read_root():
    return {"message": "API voicerecognition!"}