import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.pinecone.pinecone_setup import initialize_pinecone_index_features
from backend.pinecone.index_creation import index_creation_entries
from backend.pinecone.evaluation import start_evaluation

def main():
    features = ['mfcc', 'sc', 'rms', 'zcr'] # Features to be extracted from the audio files. Available options: 'mfcc', 'sc', 'rms', 'zcr'.
    n_mfcc = 13 # Number of MFCCs to extract if 'mfcc' is in features. Default is 13.
    dimension = 0  # Dimension of the vectors to be stored in the Pinecone index. It will be calculated based on the features.
    for feature in features:
        if feature == 'mfcc':
            dimension += 13
        elif feature == 'sc' or feature == 'rms' or feature == 'zcr':
            dimension += 1
    print(f"Dimension: {dimension}")
    
    data_percentage = 1.0 # Percentage of data to be used for indexing. Default is 1.0 (100%).
            
    feature_names_str = "-".join(features)
    indexName = f"speaker-recognition-{feature_names_str}"  # Index name to be used in Pinecone
    index = initialize_pinecone_index_features(indexName, dimension, "cosine")
    
    vectors_for_upsert, audio_speaker_count = index_creation_entries(features, n_mfcc, data_percentage)

    # Upsert the vectors into the Pinecone index
    print(f"\nStarted upserting of {len(vectors_for_upsert)} scaled vectors in the index '{indexName}'...")
    for i in range(0, len(vectors_for_upsert), 100):
        batch = vectors_for_upsert[i:i + 100]
        index.upsert(vectors=batch)
    print(f"Upserting completed. {len(vectors_for_upsert)} vectors loaded in the index '{indexName}'.")

    start_evaluation(index, features, audio_speaker_count, data_percentage)

if __name__ == "__main__":
    main()
    