import json
from pathlib import Path
from typing import Dict, Any

# --- Function to load metadata from audioMNIST_meta.txt ---
def load_metadata(metadata_path: str = "dataset/audioMNIST_meta.txt") -> Dict[str, Any]:
    base = Path(__file__).parent.parent
    full_path = (base / metadata_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"File metadati non trovato: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# --- Function to count audio files per speaker ---
def count_audio_files_per_speaker(speaker_id: str, base_folder: str = "dataset/train") -> int:
    """
    Count the number of audio files in the folder dataset/training/{speaker_id}.

    Args:
        speaker_id (str): The speaker ID (subfolder name).
        base_folder (str): The base folder path (default: "dataset/training").

    Returns:
        int: The number of audio files in the specified folder.
    """
    folder_path = Path(__file__).parent.parent / base_folder / speaker_id
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    # Count only files (not subfolders)
    return sum(1 for f in folder_path.iterdir() if f.is_file())

# --- Function to compute evaluation metrics ---
def compute_metrics(predictions: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Compute evaluation metrics: Precision@5, Accuracy (Hit@1), Recall, F1-score.

    Args:
        predictions (Dict[str, Any]): The model predictions (output of Pinecone).
        filename (str): The name of the input audio file (e.g., '0_01_4').

    Returns:
        Dict[str, Any]: A dictionary of computed metrics.
    """
    id_ground_truth = filename.split("_")[1]
    num_relevant_total = count_audio_files_per_speaker(id_ground_truth)
    print(f"Number of relevant documents for speaker {id_ground_truth}: {num_relevant_total}")

    matches = predictions.get('matches', [])
    top_k = min(5, len(matches))
    top_matches = matches[:top_k]

    # Conta i documenti rilevanti (con speaker_id corretto) nei primi 5
    relevant_retrieved = sum(1 for match in top_matches
                             if match['metadata'].get('speaker_id') == id_ground_truth)

    # Precision@5
    precision_at_5 = relevant_retrieved / top_k if top_k > 0 else 0.0

    # Recall: rilevanti recuperati / totali rilevanti per lo speaker
    recall = relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0.0

    # F1-score (con precision e recall appena calcolati)
    if precision_at_5 + recall > 0:
        f1 = 2 * (precision_at_5 * recall) / (precision_at_5 + recall)
    else:
        f1 = 0.0
    
    print(f"Metrics: \n Precision@5: {precision_at_5:.4f}, \n Recall: {recall:.4f}, \n F1-score: {f1:.4f}")

    return {
        "precision_at_5": round(precision_at_5, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
