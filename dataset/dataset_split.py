import random
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

random.seed(42)

AUDIO_DIR = Path("AudioMNIST")
TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

if not AUDIO_DIR.exists():
    raise FileNotFoundError(f"Audio directory {AUDIO_DIR} does not exist.")

# speaker_digit → list of file paths
data = defaultdict(list)

# Recursively find all .wav files in the AUDIO_DIR
for filepath in AUDIO_DIR.rglob("*.wav"):
    digit, speaker, _ = filepath.stem.split("_")
    key = (speaker, digit)
    data[key].append(filepath)

# Create train and test directories
for (speaker, digit), files in data.items():
    random.shuffle(files)
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for f in train_files:
        dest_dir = TRAIN_DIR / speaker
        dest_dir.mkdir(parents=True, exist_ok=True)
        copyfile(f, dest_dir / f.name)

    for f in test_files:
        dest_dir = TEST_DIR / speaker
        dest_dir.mkdir(parents=True, exist_ok=True)
        copyfile(f, dest_dir / f.name)

print(f"Split completed:")
print(f"  - Training files: {sum(1 for _ in TRAIN_DIR.rglob('*.wav'))}")
print(f"  - Testing files : {sum(1 for _ in TEST_DIR.rglob('*.wav'))}")
