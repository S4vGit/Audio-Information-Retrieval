import sys
import json
import numpy as np
import joblib
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
from backend.pinecone.feature_extraction import extract_combined_features_vector
from sklearn.preprocessing import StandardScaler

with open("dataset/audioMNIST_meta.txt", "r") as f:
    metadata_json = json.load(f)

def index_creation_entries(features: str, n_mfcc: int = 13, percentage: float = 1.0):
    """
    Extracts features from audio files in the training dataset, scales them, and prepares them for upsert into a Pinecone index.
    
    Args:
        features (list): List of feature types to extract from the audio files.
        n_mfcc (int): Number of MFCCs to extract if 'mfcc' is in features.
        percentage (float): Percentage of audio files to use for training, between 0.0 and 1.0.
    
    Returns:
        tuple: A tuple containing:
            - List of dictionaries with scaled feature vectors and metadata for upsert.
            - Dictionary with counts of selected audio files per speaker.
        
    """
    
    if not (0.0 <= percentage <= 1.0):
        raise ValueError("The percentage must be between 0.0 and 1.0.")
    
    raw_training_vectors = []
    training_vector_ids = []
    training_metadata_list = []

    AUDIO_DIR_TRAIN = Path("dataset/train")
    
    speaker_audio_files = defaultdict(list)
    selected_audio_counts_per_speaker = defaultdict(int)
    print(f"Grouping audio files per speaker from: {AUDIO_DIR_TRAIN}...")
    for audio_path in AUDIO_DIR_TRAIN.rglob("*.wav"):
        try:
            parts = audio_path.stem.split("_")
            if len(parts) >= 2:
                speaker_id = parts[1]
                speaker_audio_files[speaker_id].append(audio_path)
            else:
                print(f"Attention: Can't parse the speaker ID from name file: {audio_path.name}. Skipped.")
        except Exception as e:
            print(f"ERROR during processing of file {audio_path}: {e}")
    print(f"Audio files found for {len(speaker_audio_files)} speaker.")

    selected_audio_paths = []
    print(f"Selection of {percentage*100:.0f}% of audio files per speaker...")
    for speaker_id, audio_list in speaker_audio_files.items():
        if not audio_list:
            continue

        num_to_select = int(len(audio_list) * percentage)
        
        if percentage > 0 and num_to_select == 0 and len(audio_list) > 0:
            num_to_select = 1

        selected_speaker_samples = random.sample(audio_list, min(num_to_select, len(audio_list)))
        selected_audio_paths.extend(selected_speaker_samples)
        selected_audio_counts_per_speaker[speaker_id] = len(selected_speaker_samples) # update count of selected audio files per speaker
    
    print(f"Total audio files selected: {len(selected_audio_paths)}")

    for audio_path in selected_audio_paths:
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
    
    return vectors_for_upsert, selected_audio_counts_per_speaker


"""# Example usage
if __name__ == "__main__":
    features = ['mfcc']
    percentage = 1.0  # Use 100% of the data
    index_creation_entries(features, 13, percentage)"""