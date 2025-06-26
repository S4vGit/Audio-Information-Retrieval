# 1. Installa la libreria Pinecone (da shell):
#    pip install pinecone-client

import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


# Load environment variables from the .env file
load_dotenv()

# Get the Pinecone API key from environment variables
api_key = os.getenv("PINECONE_API_KEY")

def setup():
    # 2. Carica metadati e costruisci il DataFrame + profilo testuale
    def load_metadata(path="dataset/audioMNIST_meta.txt"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_profile(row):
        """parts = []
        parts.append(f"id {row['speaker_id']}")
        parts.append(f"age {row['age']}")
        parts.append(f"gender {row['gender']}")
        parts.append(f"accent {row['accent']}")
        parts.append(f"native {row['native speaker']}")
        parts.append(f"origin {row['origin']}")
        parts.append(f"recorded_in {row['recordingroom']} on {row['recordingdate']}")"""
        
        speaker_id = row['speaker_id']
        accent = str(row['accent']).strip().lower()
        age = row['age']
        gender = str(row['gender']).strip().lower()
        
        # Gestione di 'native speaker' per una frase più naturale
        native_speaker_status = "is a native speaker" if str(row['native speaker']).strip().lower() == 'yes' else "is not a native speaker"
        
        # Gestione di 'origin' per estrarre città e paese se possibile
        origin_parts = [p.strip() for p in str(row['origin']).split(',') if p.strip()]
        country = origin_parts[-1] if len(origin_parts) > 0 else "unknown country"
        city = origin_parts[-2] if len(origin_parts) > 1 else "unknown city"
        
        # Normalizza recordingroom (es. 'VR-Room' -> 'vr-room')
        recording_room = str(row['recordingroom']).strip().lower()
        
        recording_date = str(row['recordingdate']) # Data come stringa, per ora
        
        profile_text = (
            f"This speaker (ID: {speaker_id}) is a {gender} person, {age} years old. "
            f"They have a {accent} accent and {native_speaker_status}. "
            f"They are from {city} in {country}. "
            f"Their recording took place in the {recording_room} on {recording_date.replace('-', '/', 2)}." # Formato data più leggibile
        )
        
        return profile_text #"; ".join(parts)

    # load & DataFrame
    metadata = load_metadata()
    df = pd.DataFrame.from_dict(metadata, orient='index')
    df.index.name = 'speaker_id'
    df.reset_index(inplace=True)
    df['profile_text'] = df.apply(make_profile, axis=1)
    #print(df.info())
    #print(df['profile_text'].head())

    # 3. Calcola gli embedding con all-MiniLM-L6-v2
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df['profile_text'].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    #print(f"Embedding: {embeddings[0:2]}")

    index_name = "audiomnist-speakers"

    try:
        # Initialize Pinecone with the API key
        pc = Pinecone(api_key=api_key)
        print(f"Connected to Pinecone.")

        # Check if the index already exists
        if not pc.has_index(index_name):
            print(f"Index '{index_name}' not found. Creating a new index...")
            pc.create_index(
                name=index_name,
                dimension=embeddings.shape[1],  
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            print(f"Index '{index_name}' succesfuly created.")
        else:
            print(f"Index '{index_name}' already exists. Connecting to it...")

        index = pc.Index(index_name)
        print(f"Connected to index '{index_name}'.")

    except Exception as e:
        print(f"Error during the initialization or creation of the index: {e}")
        raise

    # 6. Prepara i vettori per l’upsert
    #    Ogni vettore richiede: (id, embedding, metadata_dict)
    to_upsert = []
    for sid, emb in zip(df['speaker_id'], embeddings):
        meta = metadata[sid]  # il tuo dizionario originale per eventuale filtraggio
        to_upsert.append((sid, emb.tolist(), meta))

    # 7. Carica in Pinecone
    #    suddividi in batch da ~100 per non superare i limiti di request
    batch_size = 100
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)

    print(f"✅ Caricati {len(to_upsert)} vettori in Pinecone nell'indice '{index_name}'.")
    
    
# Example usage
if __name__ == "__main__":
    #setup()
    index_name = "audiomnist-speakers"
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print(f"Connected to index '{index_name}'.")
    
    # esempio di query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(["all female speakers who recorded in Ruheraum"], convert_to_numpy=True)
    results = index.query(
        vector=query_emb.tolist()[0],
        top_k=5,
        include_metadata=True
    )
    print(results)

    
