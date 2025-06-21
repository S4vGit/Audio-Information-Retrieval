import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.pinecone.pinecone_setup import initialize_pinecone_index
from backend.pinecone.index_creation import index_creation_entries
from backend.pinecone.evaluation import start_evaluation

def main():
    features = ['mfcc', 'sc'] # Features to be extracted from the audio files. Available options: 'mfcc', 'sc', 'rms', 'zcr'.
    n_mfcc = 13 # Number of MFCCs to extract if 'mfcc' is in features. Default is 13.
    dimension = 0  # Dimension of the vectors to be stored in the Pinecone index. It will be calculated based on the features.
    for feature in features:
        if feature == 'mfcc':
            dimension += 13
        elif feature == 'sc' or feature == 'rms' or feature == 'zcr':
            dimension += 1
            
    indexName = "speaker-recognition-mfcc-sc" # Index name to be used in Pinecone
    index = initialize_pinecone_index(indexName, dimension, "cosine")
    
    vectors_for_upsert = index_creation_entries(features, n_mfcc)
    # Upsert dei vettori nell'indice Pinecone
    print(f"\nStarted upserting of {len(vectors_for_upsert)} scaled vectors in the index '{indexName}'...")
    for i in range(0, len(vectors_for_upsert), 100):
        batch = vectors_for_upsert[i:i + 100]
        index.upsert(vectors=batch)
    print(f"Upserting completed. {len(vectors_for_upsert)} vectors loaded in the index '{indexName}'.")

    start_evaluation(index, features)

if __name__ == "__main__":
    main()
    