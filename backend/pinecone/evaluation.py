import os, sys
import numpy as np
import joblib
from collections import defaultdict
from sklearn.metrics import recall_score, f1_score
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.pinecone.feature_extraction import extract_combined_features_vector
from backend.pinecone.pinecone_setup import initialize_pinecone_index

def start_evaluation(index, features: list, ):
    """
    Executes the evaluation of the Pinecone index using the specified features.
    
    Args:
        index (pinecone.Index): The Pinecone index to evaluate.
        features (list): List of feature types to use for evaluation. Supported types are
                        ('mfcc', 'sc', 'rms', 'zcr').
    """

    VECTOR_DIMENSION = 0 # Dimension of the vectors stored in the Pinecone index. It will be calculated based on the features.
    for feature in features:
        if feature == 'mfcc':
            VECTOR_DIMENSION += 13
        elif feature == 'sc' or feature == 'rms' or feature == 'zcr':
            VECTOR_DIMENSION += 1
            
    scalers_dir = Path("backend/scalers") # Directory where the scalers are stored
    feature_names_str = "_".join(features)
    scaler_filename = scalers_dir / f"scaler_feature_{feature_names_str}.pkl"
    if not scaler_filename.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_filename}. Make sure the scaler has been created and saved before running evaluation.")

    try:
        feature_scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded successfully from: {scaler_filename}")
    except Exception as e:
        raise Exception(f"Error during scaler loading from {scaler_filename}: {e}")
    
    def get_speaker_id_from_filename(file_path):
        """
        Gets the speaker ID from the filename of the audio file.
        
        Args:
            file_path (str): Path to the audio file.
            
        Returns:
            str: The speaker ID extracted from the filename, or None if it cannot be determined.
        """
        base_name = os.path.basename(file_path)
        parts = base_name.split('_')
        if len(parts) >= 2:
            return parts[1]
        return None

    def count_speakers_in_index(pinecone_index):
        """ 
        Counts the number of audio items in the Pinecone index for each speaker.

        Args:
            pinecone_index: The Pinecone index to query for speaker counts.

        Returns:
            dict: A dictionary with speaker IDs as keys and the count of their audio items as values.
        """
        speaker_counts = defaultdict(int)

        print(f"Counting items in Pinecone index '{pinecone_index.namespace}' to get speaker distribution...")

        try:
            stats = pinecone_index.describe_index_stats()
            if stats.dimension == 0 and not stats.namespaces: # Check if the index is empty
                print("Index is empty.")
                return speaker_counts

            all_possible_speaker_ids = [f"{i:02d}" for i in range(1, 61)]

            print(f"Attempting to count each speaker_id using filtered queries (can be slow for large indices)...")
            for speaker_id in all_possible_speaker_ids:
                try:
                    dummy_vector = [0.0] * VECTOR_DIMENSION 
                    speaker_query_results = pinecone_index.query(
                        vector=dummy_vector,
                        top_k=10000, # Sufficientemente grande per catturare tutti
                        filter={"speaker_id": speaker_id},
                        include_metadata=False
                    )
                    speaker_counts[speaker_id] = len(speaker_query_results.matches)

                except Exception as e:
                    print(f"Error counting speaker {speaker_id} in Pinecone: {e}")
                    speaker_counts[speaker_id] = 0
        except Exception as e:
            print(f"Error getting index stats: {e}")
        return speaker_counts

    # Compute the total number of relevant items per speaker in the index
    total_relevant_items_per_speaker = count_speakers_in_index(index)


    test_set_path = "dataset/test"

    # Lists to store true and predicted labels for different metrics
    true_labels_p1 = []
    predicted_labels_p1 = []
    true_labels_p5 = [] 
    predicted_labels_p5 = []
    true_labels_p10 = []
    predicted_labels_p10 = [] # For Recall and F1 (sklearn)

    total_queries = 0
    correct_predictions_p1 = 0
    correct_predictions_p5 = 0
    correct_predictions_p10 = 0

    retrieved_true_positives_per_speaker = defaultdict(int) # For custom Recall

    print("\nStarting evaluation on the test set...")

    for speaker_dir in os.listdir(test_set_path):
        speaker_id = speaker_dir
        speaker_full_path = os.path.join(test_set_path, speaker_dir)

        if os.path.isdir(speaker_full_path):
            for audio_file_name in os.listdir(speaker_full_path):
                if audio_file_name.endswith(".wav"):
                    file_path = os.path.join(speaker_full_path, audio_file_name)
                    true_speaker_id = get_speaker_id_from_filename(file_path)

                    if true_speaker_id is None or true_speaker_id != speaker_id:
                        print(f"Warning: Mismatch or missing speaker ID for {file_path}. Expected {speaker_id}, Got {true_speaker_id}. Skipping.")
                        continue

                    # Extracting vector features from the audio file (not scaled)
                    raw_query_vector = extract_combined_features_vector(file_path, features)

                    if raw_query_vector is None:
                        continue

                    if feature_scaler is None:
                        print("Error: StandardScaler is not initialized. Exiting evaluation.")
                        exit()

                    # Scaling the raw vector
                    scaled_query_vector = feature_scaler.transform(raw_query_vector.reshape(1, -1))[0] 

                    total_queries += 1 # Increment the total queries count

                    try:
                        # Querying Pinecone with the scaled vector
                        query_results_10 = index.query(
                            vector=scaled_query_vector.tolist(), # Scaled vector
                            top_k=10,
                            include_metadata=True
                        )

                        predicted_speaker_ids_full = [match.metadata.get('speaker_id') for match in query_results_10.matches]

                        # --- Updating Precision@1 ---
                        if len(query_results_10.matches) >= 1:
                            predicted_speaker_id_1 = query_results_10.matches[0].metadata.get('speaker_id')
                            true_labels_p1.append(true_speaker_id)
                            predicted_labels_p1.append(predicted_speaker_id_1)
                            if predicted_speaker_id_1 == true_speaker_id: # If the first match is correct
                                correct_predictions_p1 += 1 # Increment correct predictions for P@1
                        else: # If no matches, append None
                            true_labels_p1.append(true_speaker_id)
                            predicted_labels_p1.append(None)

                        # --- Updating Precision@5 ---
                        # Check if the true speaker is in the top-5 predictions
                        if len(query_results_10.matches) >= 5:
                            predicted_speaker_ids_5 = [match.metadata.get('speaker_id') for match in query_results_10.matches[:5]]
                            if true_speaker_id in predicted_speaker_ids_5: # If the true speaker is among the top-5
                                correct_predictions_p5 += 1 # Increment correct predictions for P@5
                        elif true_speaker_id in predicted_speaker_ids_full: # If there are less than 5 matches, check against all
                            correct_predictions_p5 += 1 # Increment correct predictions for P@5

                        # --- Updating Precision@10 and Recall ---
                        true_labels_p10.append(true_speaker_id)
                        predicted_labels_p10.append(predicted_speaker_ids_full)
                        if true_speaker_id in predicted_speaker_ids_full: # If the true speaker is among the top-10 predictions
                            correct_predictions_p10 += 1 # Increment correct predictions for P@10
                            retrieved_true_positives_per_speaker[true_speaker_id] += 1 # Increment true positives for custom Recall

                    except Exception as e:
                        print(f"Error querying Pinecone for {file_path}: {e}")
                        continue

    print(f"Elaboration completed. Total queries: {total_queries}")


    # --- Compute and print metrics ---

    if total_queries == 0:
        print("No processed query. Cannot compute metrics.")
    else:
        # --- Precision@k ---
        precision_1 = correct_predictions_p1 / total_queries
        precision_5 = correct_predictions_p5 / total_queries # Calculated as the number of queries where the true speaker is in the top-5 predictions divided by total queries.
        precision_10 = correct_predictions_p10 / total_queries # Calculated as the number of queries where the true speaker is in the top-10 predictions divided by total queries.

        print(f"\n--- Precision metrics (Accuracy@k) ---")
        print(f"Precision@1: {precision_1:.4f}")
        print(f"Precision@5: {precision_5:.4f}")
        print(f"Precision@10: {precision_10:.4f}")

        # --- Recall (custom)---
        individual_speaker_recall = {}
        for speaker_id in total_relevant_items_per_speaker:
            total_in_index = total_relevant_items_per_speaker[speaker_id]
            if total_in_index > 0:
                retrieved_for_this_speaker = retrieved_true_positives_per_speaker.get(speaker_id, 0)
                individual_speaker_recall[speaker_id] = retrieved_for_this_speaker / total_in_index
            else:
                individual_speaker_recall[speaker_id] = 0

        if individual_speaker_recall:
            valid_recalls = [r for r in individual_speaker_recall.values() if not np.isnan(r) and not np.isinf(r)]
            if valid_recalls:
                macro_recall = np.mean(valid_recalls)
                print(f"\n--- Recall metrics (based on the Top-10 predictions, custom) ---")
                print(f"Macro Recall (average per speaker): {macro_recall:.4f}")
            else:
                print("\nNo valid recalls found for Macro Recall (custom).")
        else:
            print("\nNo enough data to calculate Recall (custom).")

        # --- Recall and F-measure with Sklearn ---
        all_speaker_ids = sorted(list(total_relevant_items_per_speaker.keys()))
        if not all_speaker_ids:
            print("\nNo speakers found in the index. Cannot compute Recall and F-measure (sklearn).")
        else:
            y_true_binary = []
            y_pred_binary = []

            # For each query, create a binary vector for true labels and predicted labels
            for i in range(len(true_labels_p10)):
                true_speaker = true_labels_p10[i]
                predicted_speakers = predicted_labels_p10[i]

                true_one_hot = [1 if speaker_id == true_speaker else 0 for speaker_id in all_speaker_ids]
                y_true_binary.append(true_one_hot)

                # Create a binary vector for predicted speakers, where 1 indicates the speaker is predicted
                pred_one_hot = [1 if speaker_id in predicted_speakers else 0 for speaker_id in all_speaker_ids]
                y_pred_binary.append(pred_one_hot)

            y_true_binary = np.array(y_true_binary)
            y_pred_binary = np.array(y_pred_binary)

            if len(y_true_binary) > 0 and len(all_speaker_ids) > 0:
                recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
                f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)

                try:
                    recall_macro_sklearn = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
                    f1_macro_sklearn = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)

                    print(f"\n--- Sklearn metrics (Micro and Macro average) ---")
                    print(f"Recall (Micro-average): {recall_micro:.4f}")
                    print(f"F-measure (Micro-average): {f1_micro:.4f}")
                    print(f"Recall (Macro-average): {recall_macro_sklearn:.4f}")
                    print(f"F-measure (Macro-average): {f1_macro_sklearn:.4f}")
                except ValueError as ve:
                    print(f"\nCould not calculate Macro Recall/F-measure with sklearn due to ValueError: {ve}.")
                    print(f"Often occurs if some classes are missing in true/predicted labels for this batch of queries.")
                    print(f"Recall (Micro-average): {recall_micro:.4f}")
                    print(f"F-measure (Micro-average): {f1_micro:.4f}")
            else:
                print("\nNo enough data to calculate Recall and F-measure with sklearn.")
            
            
            
"""# Example usage
if __name__ == "__main__":
    index = initialize_pinecone_index("speaker-recognition-mfcc-sc", 14)
    start_evaluation(index, features=['mfcc', 'sc'])"""