import spacy
import json

nlp = spacy.load("en_core_web_trf")

metadata_path = "dataset/audioMNIST_meta.txt"

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def make_profile(id, value):
    speaker_id = id
    accent = str(value['accent']).strip().lower()
    age = value['age']
    gender = str(value['gender']).strip().lower()
    native_speaker_status = "is a native speaker" if str(value['native speaker']).strip().lower() == 'yes' else "is not a native speaker"
    origin_parts = [p.strip() for p in str(value['origin']).split(',') if p.strip()]
    country = origin_parts[-1] if len(origin_parts) > 0 else "unknown country"
    city = origin_parts[-2] if len(origin_parts) > 1 else "unknown city"
    recording_room = str(value['recordingroom']).strip().lower()
    recording_date = str(value['recordingdate'])  # Data come stringa, per ora
    profile_text = (
        f"This speaker (ID: {speaker_id}) is a {gender} from {city}, {country}. "
        f"They have a {accent} accent, are {age} years old, and {native_speaker_status}. "
        f"Recording room: {recording_room}, Date: {recording_date}."
    )
    return profile_text
# Print the first speaker's key and value
id, value = next(iter(metadata.items()))
#print(first_key, first_value)  # or just print(first_value) if you want only the data


text_profile = make_profile(id, value)

print(text_profile)
text_entity = nlp(text_profile)
print([(ent.text, ent.label_) for ent in text_entity.ents])

text = "find all male speakers with a not german accent under 30 years old"
[(ent.text, ent.label_) for ent in nlp(text).ents]

print([(ent.text, ent.label_) for ent in nlp(text).ents])

"""
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Caricamento dei Metadati
# Si assume che il file audioMNIST_meta.txt contenga il JSON fornito.
# Se il file è audioMNIST_meta.txt e contiene solo il JSON, possiamo leggerlo direttamente.
try:
    with open('dataset/audioMNIST_meta.txt', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
except json.JSONDecodeError as e:
    print(f"Errore nella decodifica JSON: {e}")
    print("Assicurati che il file contenga un JSON valido.")
    exit()
except FileNotFoundError:
    print("File 'audioMNIST_meta.txt' non trovato. Assicurati che sia nella stessa directory del codice.")
    exit()

# 2. Inizializzazione del Modello di Embedding
# Scegli un modello SentenceTransformer. 'paraphrase-MiniLM-L6-v2' è un buon punto di partenza:
# è piccolo, veloce e offre buone prestazioni per compiti di somiglianza semantica.
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 3. Preparazione dei Metadati per gli Embedding
# Creiamo una stringa descrittiva per ogni speaker che il modello possa "comprendere".
# Escludiamo i campi che non sono rilevanti per la ricerca semantica diretta (come recordingdate, age, native speaker booleano)
# o che possono essere gestiti diversamente se la query li include.
# Per l'età, potremmo considerare di convertirla in fasce (es. 'young', 'middle-aged', 'old') se volessimo includerla semanticamente.
# Per semplicità, in questo esempio ci concentriamo su accent, gender, origin e recordingroom.

speaker_descriptions = []
speaker_ids = []

for speaker_id, data in metadata.items():
    description_parts = []

    if 'gender' in data and data['gender']:
        description_parts.append(f"Il genere è {data['gender'].lower()}.")
    if 'accent' in data and data['accent']:
        description_parts.append(f"L'accento è {data['accent'].lower()}.")
    if 'origin' in data and data['origin']:
        description_parts.append(f"L'origine è {data['origin'].lower()}.")
    if 'recordingroom' in data and data['recordingroom']:
        description_parts.append(f"La stanza di registrazione è {data['recordingroom'].lower()}.")

    # Aggiungiamo anche l'età per completezza, ma va notato che il matching semantico su numeri
    # puri è meno efficace senza categorizzazione.
    if 'age' in data and str(data['age']).isdigit(): # Assicurati che l'età sia un numero
        description_parts.append(f"L'età è {data['age']} anni.")

    full_description = " ".join(description_parts)
    speaker_descriptions.append(full_description)
    speaker_ids.append(speaker_id)

# Generazione degli embedding per tutti gli speaker
print("Generazione degli embedding per gli speaker...")
speaker_embeddings = model.encode(speaker_descriptions, show_progress_bar=True)
print(f"Esempio embedding {speaker_ids[0]}: {speaker_embeddings[0]}...")  # Mostra i primi 5 valori dell'embedding del primo speaker
print("Embedding generati.")

# 4. Funzione per l'Elaborazione della Query e il Recupero
def search_speakers(query_text, top_k=5, threshold=0.5):
    print(f"\nElaborazione della query: '{query_text}'")

    # Genera l'embedding per la query
    query_embedding = model.encode(query_text)

    # Calcola la somiglianza coseno tra la query e tutti gli speaker embeddings
    # reshape(-1, 1) è necessario perché cosine_similarity si aspetta matrici 2D.
    similarities = cosine_similarity(query_embedding.reshape(1, -1), speaker_embeddings)[0]

    # Combina i punteggi di somiglianza con gli ID degli speaker
    results = []
    for i, score in enumerate(similarities):
        results.append({
            'speaker_id': speaker_ids[i],
            'description': speaker_descriptions[i],
            'similarity_score': score,
            'metadata': metadata[speaker_ids[i]] # Aggiungi i metadati originali
        })

    # Ordina i risultati per punteggio di somiglianza in ordine decrescente
    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    # Filtra i risultati basati sulla soglia e restituisci i top_k
    filtered_results = [r for r in results if r['similarity_score'] >= threshold]

    return filtered_results[:top_k]

# 5. Esempio di Utilizzo
if __name__ == "__main__":
    # Esempio 1: Speaker maschi tedeschi registrati in una stanza tipo cinema
    query1 = "female speakers with age from 20 to 30"
    results1 = search_speakers(query1, top_k=5, threshold=0.4) # Aumentata la soglia per risultati più specifici

    print(f"\nRisultati per la query: '{query1}'")
    if results1:
        for r in results1:
            print(f"  Speaker ID: {r['speaker_id']}")
            print(f"    Punteggio di Somiglianza: {r['similarity_score']:.4f}")
            print(f"    Metadati: {r['metadata']}")
            print("-" * 30)
    else:
        print("Nessun risultato trovato per questa query con la soglia specificata.")

    # Esempio 2: Donna danese
    query2 = "german speakers who recorded in vr-room"
    results2 = search_speakers(query2, top_k=3, threshold=0.3) # Soglia leggermente più bassa per meno attributi

    print(f"\nRisultati per la query: '{query2}'")
    if results2:
        for r in results2:
            print(f"  Speaker ID: {r['speaker_id']}")
            print(f"    Punteggio di Somiglianza: {r['similarity_score']:.4f}")
            print(f"    Metadati: {r['metadata']}")
            print("-" * 30)
    else:
        print("Nessun risultato trovato per questa query con la soglia specificata.")

    # Esempio 3: Speaker con accento spagnolo
    query3 = "spanish accent speakers"
    results3 = search_speakers(query3, top_k=5, threshold=0.4)

    print(f"\nRisultati per la query: '{query3}'")
    if results3:
        for r in results3:
            print(f"  Speaker ID: {r['speaker_id']}")
            print(f"    Punteggio di Somiglianza: {r['similarity_score']:.4f}")
            print(f"    Metadati: {r['metadata']}")
            print("-" * 30)
    else:
        print("Nessun risultato trovato per questa query con la soglia specificata.")

    # Esempio 4: Speaker molto giovane
    query4 = "speakers with age under 30"
    results4 = search_speakers(query4, top_k=3, threshold=0.2) # Soglia molto bassa per l'età numerica

    print(f"\nRisultati per la query: '{query4}'")
    if results4:
        for r in results4:
            print(f"  Speaker ID: {r['speaker_id']}")
            print(f"    Punteggio di Somiglianza: {r['similarity_score']:.4f}")
            print(f"    Metadati: {r['metadata']}")
            print("-" * 30)
    else:
        print("Nessun risultato trovato per questa query con la soglia specificata.")
"""