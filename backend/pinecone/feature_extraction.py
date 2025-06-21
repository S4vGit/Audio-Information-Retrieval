import librosa
import numpy as np

def extract_mfcc_vector(file_path, n_mfcc=13):
    """
    Extracts the mean MFCC vector from an audio file.

    Args:
        file_path (_type_): Path to the audio file.
        n_mfcc (int, optional): Number of MFCCs to extract. Defaults to 13.

    Returns:
        np.ndarray: Mean MFCC vector of shape (n_mfcc,).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            print(f"Warning: Audio file {file_path} is empty or too short. Skipping MFCC extraction.")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1)
    
    except Exception as e:
        print(f"Error processing {file_path} for SC: {e}")
        return None

def extract_spectral_centroid_vector(file_path):
    """
    Extracts the mean Spectral Centroid from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        np.ndarray: Mean Spectral Centroid vector of shape (1,).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            print(f"Warning: Audio file {file_path} is empty or too short. Skipping SC extraction.")
            return None
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        return np.array([np.mean(spectral_centroids)]) # Ritorna un array numpy di dimensione (1,)
    except Exception as e:
        print(f"Error processing {file_path} for SC: {e}")
        return None

def extract_rms_vector(file_path):
    """
    Extracts the Root Mean Square energy from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        np.ndarray: Mean RMS vector of shape (1,).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            print(f"Warning: Audio file {file_path} is empty or too short. Skipping RMS extraction.")
            return None
        rms = librosa.feature.rms(y=y)[0]
        return np.array([np.mean(rms)])
    except Exception as e:
        print(f"Error processing {file_path} for RMS: {e}")
        return None

def extract_zcr_vector(file_path):
    """
    Extracts the Zero Crossing Rate from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        np.ndarray: Mean ZCR vector of shape (1,).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            print(f"Warning: Audio file {file_path} is empty or too short. Skipping ZCR extraction.")
            return None
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        return np.array([np.mean(zcr)])
    except Exception as e:
        print(f"Error processing {file_path} for ZCR: {e}")
        return None

def extract_combined_features_vector(file_path, feature_types, n_mfcc=13):
    """
    Extracts a combined feature vector from an audio file based on specified feature types.
    
    Args:
        file_path (str): Path to the audio file.
        feature_types (list): List of feature types to extract. Supported types are
                               ('mfcc', 'sc', 'rms', 'zcr').
        n_mfcc (int): Number of MFCCs to extract if 'mfcc' is in feature_types.
        
    Returns:
        np.ndarray: Combined feature vector containing the extracted features.
    """
    all_features = []
    for f_type in feature_types:
        feature_vector = None
        if f_type == 'mfcc':
            feature_vector = extract_mfcc_vector(file_path, n_mfcc)
        elif f_type == 'sc':
            feature_vector = extract_spectral_centroid_vector(file_path)
        elif f_type == 'rms':
            feature_vector = extract_rms_vector(file_path)
        elif f_type == 'zcr':
            feature_vector = extract_zcr_vector(file_path)
        else:
            print(f"Warning: Unknown feature type '{f_type}'. Skipping.")

        if feature_vector is not None:
            all_features.append(feature_vector)
        else:
            return None # If any feature extraction fails, return None

    if not all_features:
        return None # If no features were extracted, return None

    # Concatenate all feature vectors into a single vector
    return np.concatenate(all_features)

"""# Example usage
if __name__ == "__main__":
    audio_path = "dataset/test/01/0_01_1.wav"
    
    features = ['mfcc']
    vector = extract_combined_features_vector(audio_path, features, 13)
    print (f"Extracted vector for {audio_path} with feature {features}: {vector}")
    
    features = ['sc']
    vector = extract_combined_features_vector(audio_path, features)
    print (f"Extracted vector for {audio_path} with feature {features}: {vector}")
    
    features = ['rms']
    vector = extract_combined_features_vector(audio_path, features)
    print (f"Extracted vector for {audio_path} with feature {features}: {vector}")
    
    features = ['zcr']
    vector = extract_combined_features_vector(audio_path, features)
    print (f"Extracted vector for {audio_path} with feature {features}: {vector}")
    
    features = ['mfcc', 'sc']
    vector = extract_combined_features_vector(audio_path, features, 13)
    print (f"Extracted vector for {audio_path} with feature {features}: {vector}")"""
    
    
    