import sys
import json
import numpy as np
import joblib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.pinecone.feature_extraction import extract_combined_features_vector
from sklearn.preprocessing import StandardScaler

with open("dataset/audioMNIST_meta.txt", "r") as f:
    metadata_json = json.load(f)

def index_creation_entries(features: str, n_mfcc: int = 13):
    """
    Extracts features from audio files in the training dataset, scales them, and prepares them for upsert into a Pinecone index.
    
    Args:
        features (list): List of feature types to extract from the audio files.
        n_mfcc (int): Number of MFCCs to extract if 'mfcc' is in features.
    
    Returns:
        list: A list of dictionaries, each containing an ID, scaled feature values, and metadata for upsert into Pinecone.
    """
    raw_training_vectors = []
    training_vector_ids = []
    training_metadata_list = []

    AUDIO_DIR_TRAIN = Path("dataset/train")

    for audio_path in AUDIO_DIR_TRAIN.rglob("*.wav"):
        try:
            # Extract combined features vector (not scaled)
            vector = extract_combined_features_vector(audio_path, features, n_mfcc)
            if vector is None:
                continue

            # Preparing metadata
            digit, speaker_id, index_id = audio_path.stem.split("_")
            current_id = audio_path.stem
            base_metadata = metadata_json.get(speaker_id, {})
            metadata = dict(base_metadata)
            metadata.update({
                "digit": int(digit),
                "speaker_id": speaker_id,
                "filename": audio_path.stem
            })

            raw_training_vectors.append(vector)
            training_vector_ids.append(current_id)
            training_metadata_list.append(metadata)

        except Exception as e:
            print(f"ERROR FROM FILE {audio_path}: {e}")

    if not raw_training_vectors:
        print("No vectors found. Exiting...")
        exit()

    raw_training_vectors_np = np.vstack(raw_training_vectors)

    # Scaler initialization and fitting
    feature_scaler = StandardScaler()
    print(f"Fitting StandardScaler on {raw_training_vectors_np.shape[0]} vectors...")
    feature_scaler.fit(raw_training_vectors_np)
    print("Scaler successfully fitted.")
    
    # Saving the scaler
    scalers_dir = Path("backend/scalers")
    scalers_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    feature_names_str = "_".join(features)
    scaler_filename = scalers_dir / f"scaler_feature_{feature_names_str}.pkl" # Name of the scaler file based on features

    joblib.dump(feature_scaler, scaler_filename)
    print(f"Scaler saved in: {scaler_filename}")
    

    # Scaling the raw training vectors
    scaled_training_vectors_np = feature_scaler.transform(raw_training_vectors_np)
    
    vectors_for_upsert = []
    for i in range(len(scaled_training_vectors_np)):
        vectors_for_upsert.append({
            "id": training_vector_ids[i],
            "values": scaled_training_vectors_np[i].tolist(),
            "metadata": training_metadata_list[i]
        })
        
    print(f"Number of scaled vectors: {len(vectors_for_upsert)}")
    
    return vectors_for_upsert


"""# Example usage
if __name__ == "__main__":
    features = ['mfcc']
    index_creation_entries(features, 13)"""