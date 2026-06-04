"""
EEG Preprocessing Script for BCI Competition IV-2a Dataset
Author: Safid Hasan
Description:
    Loads, filters, epochs, and saves EEG data for motor imagery classification.
    Output format: X (EEG trials), y (labels)
"""

import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = r"C:\Users\safid\Downloads\EEGNet\data"   # <-- change to your dataset folder
SUBJECT = "A02T.gdf"                  # Example subject file
SAVE_PATH = r"C:\Users\safid\Downloads\EEGNet\preprocessed_data"
TMIN, TMAX = 2.0, 6.0                 # Time window (in seconds)
LOW_FREQ, HIGH_FREQ = 4.0, 40.0       # Bandpass range
TEST_SIZE = 0.3                       # 70/30 train-test split
RANDOM_SEED = 42

# -----------------------------
# STEP 1: LOAD RAW DATA
# -----------------------------
print(f"Loading subject data: {SUBJECT}")
raw = mne.io.read_raw_gdf(os.path.join(DATA_PATH, SUBJECT), preload=True)

# -----------------------------
# STEP 2: FILTER (4–40 Hz)
# -----------------------------
print("Applying bandpass filter (4–40 Hz)...")
raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design='firwin')

# -----------------------------
# STEP 3: SELECT EEG CHANNELS
# -----------------------------
print("Selecting EEG channels only...")
eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG') or ch[0] in ['F','C','P','O']]
raw.pick_channels(eeg_channels)
print(f"Channels retained: {len(raw.ch_names)}")

# -----------------------------
# STEP 4: SET COMMON AVERAGE REFERENCE
# -----------------------------
print("Applying common average reference (CAR)...")
raw.set_eeg_reference('average', projection=True)

# -----------------------------
# -----------------------------
# STEP 5: EXTRACT EVENTS & EPOCHS (FIXED)
print("Extracting events and creating epochs (2–6s)...")
events, event_id = mne.events_from_annotations(raw)

# Filter only motor imagery event codes (left, right, feet, tongue)
mi_event_ids = {'769': event_id.get('769'),
                '770': event_id.get('770'),
                '771': event_id.get('771'),
                '772': event_id.get('772')}
mi_event_ids = {k: v for k, v in mi_event_ids.items() if v is not None}

epochs = mne.Epochs(
    raw,
    events,
    event_id=mi_event_ids,
    tmin=TMIN,
    tmax=TMAX,
    baseline=None,
    preload=True,
    on_missing='ignore',
    event_repeated='drop'
)
# -----------------------------
# STEP 6: CONVERT TO NUMPY
# -----------------------------
print("Converting epochs to NumPy arrays...")
X = epochs.get_data()           # shape: (n_trials, n_channels, n_samples)
y = epochs.events[:, -1]        # labels

print(f"Data shape: X={X.shape}, y={y.shape}")

# -----------------------------
# STEP 7: TRAIN-TEST SPLIT
# -----------------------------
print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# -----------------------------
# STEP 8: SAVE OUTPUTS
# -----------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
np.save(os.path.join(SAVE_PATH, f"{SUBJECT[:-4]}_X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, f"{SUBJECT[:-4]}_X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, f"{SUBJECT[:-4]}_y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, f"{SUBJECT[:-4]}_y_test.npy"), y_test)

print("✅ Preprocessing complete!")
print(f"Saved files in: {SAVE_PATH}")
print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")