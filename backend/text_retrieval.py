# backend/utils/ir_utils.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Funzione per caricare i metadati dal file JSON
def load_metadata(metadata_path="dataset/audioMNIST_meta.txt"):
    """
    Carica i metadati degli speaker dal file audioMNIST_meta.txt.
    Args:
        metadata_path (str): Il percorso del file dei metadati.
    Returns:
        dict: Un dizionario contenente i metadati degli speaker.
    """
    
    # Un modo robusto per trovare il file indipendentemente dalla directory corrente
    current_dir = Path(__file__).parent
    
    #print(current_dir.parent.parent)
    
    # Se 'audioMNIST_meta.txt' è nella root del progetto e questo script è in 'backend/utils/'
    metadata_file_path =  current_dir.parent / metadata_path #current_dir.parent.parent /

    if not metadata_file_path.exists():
        raise FileNotFoundError(f"File metadati non trovato: {metadata_file_path}")

    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Funzione per costruire profili testuali leggibili dagli embedding
def build_textual_profiles(metadata):
    """
    Costruisce un profilo testuale per ogni speaker unendo i suoi attributi.
    Args:
        metadata (dict): I metadati degli speaker.
    Returns:
        dict: Un dizionario dove la chiave è l'ID dello speaker e il valore è il suo profilo testuale.
    """
    profiles = {}
    for speaker_id, attributes in metadata.items():
        # Costruisce una stringa unendo tutti gli attributi del speaker.
        # Ad esempio: "accent: german age: 30 gender: male native speaker: no ..."
        profile_text = " ".join([f"{k}: {v}" for k, v in attributes.items()])
        profiles[speaker_id] = profile_text
    return profiles

# Funzione per calcolare gli embedding dei profili testuali
def compute_embeddings(profiles, model):
    """
    Calcola gli embedding per i profili testuali usando un modello SentenceTransformer.
    Args:
        profiles (dict): Dizionario di profili testuali (ID speaker -> testo).
        model (SentenceTransformer): Il modello pre-addestrato per gli embedding.
    Returns:
        tuple: Una tupla contenente (lista degli ID speaker, array numpy degli embedding).
    """
    speaker_ids = list(profiles.keys())
    texts = list(profiles.values())
    embeddings = model.encode(texts, convert_to_numpy=True)
    return speaker_ids, embeddings

# Blocco principale per l'esecuzione del codice (esempio di utilizzo)
if __name__ == "__main__":
    print("--- Inizio del processo di embedding e similarità ---")

    # 1. Caricamento dei metadati
    try:
        metadata = load_metadata()
        print(f"Caricati metadati per {len(metadata)} speaker.")
    except FileNotFoundError as e:
        print(e)
        print("Assicurati che il file 'audioMNIST_meta.txt' sia nel percorso corretto rispetto a questo script.")
        print("Ad esempio, se lo script è in 'backend/utils/' e 'audioMNIST_meta.txt' è in 'dataset/',")
        print("assicurati che il percorso nella funzione load_metadata sia '../../dataset/audioMNIST_meta.txt'.")
        exit() # Esce se il file non viene trovato

    # 2. Costruzione dei profili testuali
    profiles = build_textual_profiles(metadata)
    print(f"Costruiti profili testuali per {len(profiles)} speaker.")

    # 3. Inizializzazione del modello SentenceTransformer
    print("Caricamento del modello SentenceTransformer 'all-MiniLM-L6-v2'...")
    # Questo passaggio potrebbe richiedere tempo la prima volta per scaricare il modello
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Modello caricato.")

    # 4. Calcolo degli embedding per i metadati
    print("Calcolo degli embedding per i metadati degli speaker...")
    speaker_ids, metadata_embeddings = compute_embeddings(profiles, text_encoder)
    print(f"Calcolati {len(metadata_embeddings)} embedding di dimensione {metadata_embeddings.shape[1]}.")
    
    # 5. Definizione e embedding di una query di esempio
    query_string = "all spanish people younger than 28 years old who recorded in vr-room"
    print(f"\nQuery di esempio: '{query_string}'")
    query_embedding = text_encoder.encode(query_string, convert_to_numpy=True)
    print("Embedding della query calcolato.")
    
    # 6. Calcolo della similarità (cosine similarity)
    # Si usa il prodotto scalare (@) per calcolare il numeratore per tutti i vettori in modo efficiente.
    # np.linalg.norm calcola la norma (lunghezza) dei vettori.
    # Aggiungiamo 1e-10 al denominatore per evitare divisioni per zero nel caso di vettori nulli.
    print("Calcolo delle similarità coseno...")
    similarities = metadata_embeddings @ query_embedding / (
        np.linalg.norm(metadata_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )

    # 7. Ordinamento e visualizzazione dei top risultati
    top_k = 5
    # argsort() restituisce gli indici che ordinerebbero l'array.
    # [-top_k:] prende gli indici dei 'top_k' valori più alti.
    # [::-1] li inverte per averli in ordine decrescente di similarità.
    top_indices = similarities.argsort()[-top_k:][::-1]

    print(f"\nTop {top_k} speaker più simili alla query:")
    results = []
    for idx in top_indices:
        speaker_id = speaker_ids[idx]
        similarity_score = similarities[idx]
        profile_text = profiles[speaker_id] # Recupera il testo del profilo per mostrare più dettagli

        print(f"  Speaker ID: {speaker_id}, Similarità: {similarity_score:.4f}")
        print(f"    Profilo: {profile_text}\n")
        
        results.append({
            "speaker_id": speaker_id,
            "similarity": float(similarity_score),
            "profile_text": profile_text,
            "metadata": metadata[speaker_id] # Include anche i metadati originali
        })
    
    print("--- Processo completato ---")
    

### SOLO CONVERSIONE E SIMILARITÀ COSENO ### 
"""# backend/utils/ir_utils.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

def load_metadata(metadata_path="dataset/audioMNIST_meta.txt"):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_textual_profiles(metadata):
    profiles = {}
    for speaker_id, attributes in metadata.items():
        profile_text = " ".join([f"{k}: {v}" for k, v in attributes.items()])
        profiles[speaker_id] = profile_text
    return profiles

def compute_embeddings(profiles, model):
    speaker_ids = list(profiles.keys())
    texts = list(profiles.values())
    embeddings = model.encode(texts, convert_to_numpy=True)
    return speaker_ids, embeddings


# Example usage
if __name__ == "__main__":
    metadata = load_metadata()
    profiles = build_textual_profiles(metadata)
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    speaker_ids, metadata_embeddings = compute_embeddings(profiles, text_encoder)
    
    # Embed della query
    query_embedding = text_encoder.encode("italian people", convert_to_numpy=True)
    
    # Calcola similarità (cosine similarity)
    similarities = metadata_embeddings @ query_embedding / (
        np.linalg.norm(metadata_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )

    # Ordina e prendi i top-5
    top_k = 5
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        print(f"Speaker ID: {speaker_ids[idx]}, Similarity: {similarities[idx]:.4f}")"""


### SOLR ###
"""def main():        
    import pysolr

    solr = pysolr.Solr('http://localhost:8983/solr/audio_metadata_core', always_commit=True)

    results = solr.search('accent:Italian AND age:[21 TO 30]')

    print(f"Saw result(s). {len(results)}")


    for result in results:
        print(f"ID: {result['id']}, Accent: {result['accent']}, Age: {result['age']}")
        
        
if __name__ == "__main__":
    main()"""
        
"""import pandas as pd
import spacy
import torch
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
#from datasets import Dataset
import ast

def load_spacy_model():
    if torch.cuda.is_available():
        spacy.require_gpu()
    return spacy.load("en_core_web_trf")
nlp = load_spacy_model()

def extract_entities_from_metadata(text):
    metadata = load_metadata()
    entities_per_speaker = {}
    # Estrae entità e le formatta come stringa: "[['entity', 'type'], ...]"
    doc = nlp(text)
    for speaker_id, attributes in metadata.items():
        # Unisci tutti i valori degli attributi in un unico testo
        text = " ".join(str(v) for v in attributes.values())
        doc = nlp(text)
        entities = [[ent.text, ent.label_] for ent in doc.ents]
        entities_per_speaker[speaker_id] = entities
    return entities_per_speaker

entities = extract_entities_from_metadata("dataset/audioMNIST_meta.txt")
print(entities)"""