import json
import pysolr
from pathlib import Path

# Connettiti al tuo core Solr
SOLR_URL = 'http://localhost:8983/solr/audio_metadata_core'
solr = pysolr.Solr(SOLR_URL, timeout=10)

def load_metadata(metadata_path="dataset/audioMNIST_meta.txt"):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def index_metadata_to_solr(metadata):
    documents = []
    for speaker_id, attributes in metadata.items():
        # Clona gli attributi per evitare di modificare il dizionario originale
        doc = attributes.copy()

        # Imposta l'ID del documento Solr
        doc['id'] = speaker_id 

        # Rinomina il campo con lo spazio se necessario (come suggerito nello schema)
        if 'native speaker' in doc:
            doc['native_speaker'] = doc.pop('native speaker')

        # Converte l'età a intero se è una stringa, Solr gestirà i tipi numerici
        if 'age' in doc and isinstance(doc['age'], str):
            try:
                doc['age'] = int(doc['age'])
            except ValueError:
                print(f"Warning: Could not convert age for speaker {speaker_id}: {doc['age']}")
                doc['age'] = None # O gestisci come preferisci

        # Crea un campo 'profile_text' per la ricerca full-text se lo desideri
        doc['profile_text'] = " ".join([f"{k}: {v}" for k, v in attributes.items()])

        documents.append(doc)

    # Aggiungi i documenti a Solr in batch
    print(f"Indicizzando {len(documents)} documenti in Solr...")
    solr.add(documents)
    print("Indicizzazione completata. Commit in corso...")
    solr.commit() # Esegui il commit per rendere i dati ricercabili
    print("Commit completato.")

if __name__ == "__main__":
    # Assicurati che la directory 'dataset' esista e contenga 'audioMNIST_meta.txt'
    Path("dataset").mkdir(exist_ok=True) 
    # Per semplicità, scrivi un meta.txt di esempio se non esiste (da rimuovere in produzione)
    """if not Path("dataset/audioMNIST_meta.txt").exists():
        print("Creazione di un file audioMNIST_meta.txt di esempio.")
        example_metadata = {
            "01": {"accent": "german", "age": 30, "gender": "male", "native speaker": "no", "origin": "Europe, Germany, Wuerzburg", "recordingdate": "17-06-22-11-04-28", "recordingroom": "Kino"},
            "02": {"accent": "German", "age": "25", "gender": "male", "native speaker": "no", "origin": "Europe, Germany, Hamburg", "recordingdate": "17-06-26-17-57-29", "recordingroom": "Kino"},
            "03": {"accent": "German", "age": "31", "gender": "female", "native speaker": "yes", "origin": "Europe, Germany, Berlin", "recordingdate": "17-06-30-17-34-51", "recordingroom": "Kino"}
        }
        with open("dataset/audioMNIST_meta.txt", 'w', encoding='utf-8') as f:
            json.dump(example_metadata, f, indent=4)"""


    metadata = load_metadata()
    index_metadata_to_solr(metadata)
    print("Dati pronti per la ricerca in Solr!")