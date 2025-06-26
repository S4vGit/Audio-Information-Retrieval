"""# backend/utils/ir_utils.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# Funzione per caricare i metadati dal file JSON
def load_metadata(metadata_path="dataset/audioMNIST_meta.txt"):    
    # Un modo robusto per trovare il file indipendentemente dalla directory corrente
    current_dir = Path(__file__).parent
    
    #print(current_dir.parent.parent)
    
    # Se 'audioMNIST_meta.txt' è nella root del progetto e questo script è in 'backend/utils/'
    metadata_file_path =  current_dir.parent / metadata_path #current_dir.parent.parent /

    if not metadata_file_path.exists():
        raise FileNotFoundError(f"File metadati non trovato: {metadata_file_path}")

    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data"""
    
    
### NON DECOMMENTARE ###
"""# Funzione per costruire profili testuali leggibili dagli embedding
def build_textual_profiles(metadata):
    profiles = {}
    for speaker_id, attributes in metadata.items():
        # Costruisce una stringa unendo tutti gli attributi del speaker.
        # Ad esempio: "accent: german age: 30 gender: male native speaker: no ..."
        profile_text = " ".join([f"{k}: {v}" for k, v in attributes.items()])
        profiles[speaker_id] = profile_text
    return profiles"""

"""
def build_textual_profiles(metadata: Dict[str, Any]) -> Dict[str, str]:
    profiles = {}
    for speaker_id, attributes in metadata.items():
        description_parts = []

        # Aggiungi una frase introduttiva più descrittiva per il gender e l'età
        gender = attributes.get('gender')
        age = attributes.get('age')
        
        if gender == "female":
            noun = "She"
        else:
            noun = "He"
        
        description_parts.append(f"The speaker {speaker_id} is a {gender} person who is {age} years old.")

        # Aggiungi informazioni sull'accento in modo più fluente
        accent = attributes.get('accent')
        if accent:
            description_parts.append(f"{noun} speaks with a {accent} accent.")

        # Aggiungi informazioni sulla madrelingua
        native_speaker = attributes.get('native_speaker')
        if native_speaker == 'yes':
            description_parts.append(f"{noun} is a native speaker.")
        elif native_speaker == 'no':
            description_parts.append(f"{noun} is not a native speaker.")

        # Aggiungi informazioni sull'origine
        origin = attributes.get('origin')
        if origin:
            description_parts.append(f"{noun} comes from {origin}.")
        
        # Aggiungi informazioni sulla stanza (se 'room' è un attributo rilevante)
        room = attributes.get('recordingroom')
        recording_date = attributes.get('recordingdate')
        description_parts.append(f"The audio was recorded in a {room} in {recording_date}.")

        # Combina le parti in una singola stringa.
        # Usa un punto o una virgola per separare le frasi per una maggiore leggibilità.
        profile_text = " ".join(description_parts).strip().lower()
        profiles[speaker_id] = profile_text if profile_text else f"Speaker ID: {speaker_id}" # Fallback
    return profiles

# Funzione per calcolare gli embedding dei profili testuali
def compute_embeddings(profiles, model):
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
    print(f"Primi 3 profili: {list(profiles.items())[:3]}")

    # 3. Inizializzazione del modello SentenceTransformer
    print("Caricamento del modello SentenceTransformer 'all-MiniLM-L6-v2'...")
    # Questo passaggio potrebbe richiedere tempo la prima volta per scaricare il modello
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Modello caricato.")

    # 4. Calcolo degli embedding per i metadati
    print("Calcolo degli embedding per i metadati degli speaker...")
    speaker_ids, metadata_embeddings = compute_embeddings(profiles, text_encoder)
    print(f"Calcolati {len(metadata_embeddings)} embedding di dimensione {metadata_embeddings.shape[1]}.")
    print(f"Primi 3 embedding: {metadata_embeddings[:3]}")
    
    # 5. Definizione e embedding di una query di esempio
    query_string = "all german speakers aged 26 to 30"
    print(f"\nQuery di esempio: '{query_string}'")
    query_embedding = text_encoder.encode(query_string, convert_to_numpy=True)
    print(f"Embedding della query calcolato.")
    #print(f"Embedding della query calcolato. {query_embedding}")
    
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
    
    print("--- Processo completato ---")"""
    
### CHAT GPT ###    
"""import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from typing import Dict, Any, List, Tuple

# -----------------------------
# 1. METADATA LOADING & NORMALIZATION
# -----------------------------

def load_metadata(metadata_path: str = "dataset/audioMNIST_meta.txt") -> Dict[str, Any]:
    base = Path(__file__).parent.parent
    full_path = (base / metadata_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"File metadati non trovato: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_date(raw: str) -> str:
    try:
        dt = datetime.strptime(raw, "%y-%m-%d-%H-%M-%S")
        return dt.strftime("%d %B %Y")
    except Exception:
        return raw


def normalize_location(raw: str) -> str:
    parts = [p.strip().title() for p in raw.split(',')]
    if len(parts) >= 2:
        city, country = parts[-1], parts[-2]
        return f"{city}, {country}"
    return ", ".join(parts)

# -----------------------------
# 2. COSTRUZIONE PROFILI TESTUALI
# -----------------------------

def build_textual_profiles(metadata: Dict[str, Any]) -> Dict[str, str]:
    profiles = {}
    for sid, attrs in metadata.items():
        sentences = []
        # gender & age
        gender = attrs.get('gender', 'unknown')
        age = attrs.get('age', 'unknown')
        sentences.append(f"Speaker {sid} is a {age}-year-old {gender}.")
        # accent
        accent = attrs.get('accent')
        if accent:
            sentences.append(f"Speaks with a {accent} accent.")
        # native
        ns = attrs.get('native_speaker')
        if ns == 'yes':
            sentences.append("Is a native speaker.")
        elif ns == 'no':
            sentences.append("Not a native speaker.")
        # location
        origin = attrs.get('origin')
        if origin:
            loc = normalize_location(origin)
            sentences.append(f"From {loc}.")
        # recording info
        room = attrs.get('recordingroom', '')
        date = normalize_date(attrs.get('recordingdate', ''))
        if room and date:
            sentences.append(f"Recorded in a {room} on {date}.")
        profiles[sid] = " ".join(sentences)
    return profiles

# -----------------------------
# 3. EMBEDDINGS & SIMPLE INDEX
# -----------------------------

def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True)
    # normalizza vettori
    faiss.normalize_L2(embs)
    return embs


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    # IndexFlatIP con inner product su vettori normalizzati = cosine similarity
    return faiss.IndexFlatIP(embs.shape[1])

# -----------------------------
# 4. TWO-STAGE RETRIEVAL
# -----------------------------

def two_stage_retrieve(
    query: str,
    speaker_ids: List[str],
    profiles: Dict[str, str],
    bi_encoder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    top_k: int = 5,
    candidates: int = 50,
    re_ranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) -> List[Tuple[str, float]]:
    # embed & normalize query
    q_emb = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # stage 1: retrieve
    D, I = index.search(q_emb, candidates)
    cids = []
    for idx in I[0]:
        sid = speaker_ids[idx]
        if sid not in cids:
            cids.append(sid)
    # limit candidates
    cids = cids[:candidates]

    # stage 2: re-ranking
    cross = CrossEncoder(re_ranker_model)
    pairs = [(query, profiles[sid]) for sid in cids]
    scores = cross.predict(pairs)
    ranked = sorted(zip(cids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

# -----------------------------
# USO DI ESEMPIO
# -----------------------------
if __name__ == "__main__":
    meta = load_metadata()
    profs = build_textual_profiles(meta)
    speaker_ids = list(profs.keys())

    bi = SentenceTransformer('all-MiniLM-L6-v2')
    texts = list(profs.values())
    embs = compute_embeddings(texts, bi)

    idx = build_faiss_index(embs)
    idx.add(embs)

    q = "all german speakers aged  to 30 to 40"
    results = two_stage_retrieve(q, speaker_ids, profs, bi, idx)
    for sid, score in results:
        print(f"{sid}: {score:.4f} -> {profs[sid]}")"""


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
        
        
### CHAT GPT 2 ###
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import re
from typing import Dict, Any, List, Tuple

# -----------------------------
# 1. METADATA LOADING & NORMALIZATION
# -----------------------------

def load_metadata(metadata_path: str = "dataset/audioMNIST_meta.txt") -> Dict[str, Any]:
    base = Path(__file__).parent.parent
    full_path = (base / metadata_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"File metadati non trovato: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_date(raw: str) -> str:
    try:
        dt = datetime.strptime(raw, "%y-%m-%d-%H-%M-%S")
        return dt.strftime("%d %B %Y")
    except Exception:
        return raw


def normalize_location(raw: str) -> str:
    parts = [p.strip().title() for p in raw.split(',')]
    if len(parts) >= 2:
        city, country = parts[-1], parts[-2]
        return f"{city}, {country}"
    return ", ".join(parts)

# -----------------------------
# 2. COSTRUZIONE PROFILI TESTUALI
# -----------------------------

def build_textual_profiles(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Costruisce un profilo testuale per ogni speaker arricchito con tutti gli attributi.
    """
    profiles = {}
    for sid, attrs in metadata.items():
        fields = []
        fields.append(f"id {sid}")
        if 'age' in attrs:
            fields.append(f"age {attrs['age']}")
        if 'gender' in attrs:
            fields.append(f"gender {attrs['gender']}")
        if 'accent' in attrs:
            fields.append(f"accent {attrs['accent']}")
        if 'native_speaker' in attrs:
            fields.append(f"native {attrs['native_speaker']}")
        if 'origin' in attrs:
            loc = normalize_location(attrs['origin'])
            fields.append(f"origin {loc}")
        if 'recordingroom' in attrs and 'recordingdate' in attrs:
            date = normalize_date(attrs['recordingdate'])
            fields.append(f"recorded_in {attrs['recordingroom']} on {date}")
        # profilo composto da keyword:value per semplificare embedding
        profiles[sid] = "; ".join(fields)
    return profiles

# -----------------------------
# 3. QUERY EXPANSION
# -----------------------------

def expand_query(query: str) -> str:
    """
    Estrae attributi dalla query e genera una descrizione espansa in linguaggio naturale.
    """
    parts = []
    # accents
    m_in = re.search(r"all ([a-zA-Z]+) speakers", query)
    m_out = re.search(r"all non-([a-zA-Z]+) speakers", query)
    if m_in:
        parts.append(f"accent {m_in.group(1)}")
    if m_out:
        parts.append(f"accent not {m_out.group(1)}")
    # age
    m_age = re.search(r"aged? (\d+) to (\d+)", query)
    if m_age:
        parts.append(f"age between {m_age.group(1)} and {m_age.group(2)}")
    # crea descrizione
    if parts:
        return query + ": " + ", ".join(parts)
    return query

# -----------------------------
# 4. EMBEDDINGS & INDEX
# -----------------------------

def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    return embs


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(embs.shape[1])

# -----------------------------
# 5. TWO-STAGE RETRIEVAL (EMBEDDING-ONLY)
# -----------------------------

def two_stage_retrieve(
    query: str,
    speaker_ids: List[str],
    profiles: Dict[str, str],
    bi_encoder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    top_k: int = 5,
    candidates: int = 50,
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) -> List[Tuple[str, float]]:
    # espansione query
    q_expanded = expand_query(query)
    # stage 1: bi-encoder retrieval
    q_emb = bi_encoder.encode([q_expanded], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, candidates)
    cids = [speaker_ids[i] for i in I[0]]
    # stage 2: cross-encoder re-ranking
    re_ranker = CrossEncoder(reranker_model)
    pairs = [(q_expanded, profiles[sid]) for sid in cids]
    scores = re_ranker.predict(pairs)
    ranked = sorted(zip(cids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

# -----------------------------
# USO DI ESEMPIO
# -----------------------------
if __name__ == "__main__":
    metadata = load_metadata()
    profiles = build_textual_profiles(metadata)
    ids = list(profiles.keys())

    bi = SentenceTransformer('thenlper/gte-base')
    texts = list(profiles.values())
    embeddings = compute_embeddings(texts, bi)

    idx = build_faiss_index(embeddings)
    idx.add(embeddings)

    tests = [
        "all german speakers aged 30 to 40",
        "all asian speakers aged 21 to 30 who recorded in a studio",
    ]
    for q in tests:
        print(f"Query: {q}")
        res = two_stage_retrieve(q, ids, profiles, bi, idx)
        for sid, score in res:
            print(f" {sid}: {score:.4f} -> {profiles[sid]}")
        print()



