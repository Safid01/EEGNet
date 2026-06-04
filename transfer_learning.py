"""
Transfer Learning Script for EEGNet (Keras 3 Compatible)
Author: Safid Hasan
Description:
    Loads a pretrained EEGNet model (.h5 or .keras) and fine-tunes it
    on another subject's EEG data for cross-subject transfer learning.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer  # for .keras/.SavedModel inference support

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = r"C:\Users\safid\Downloads\EEGNet\preprocessed_data"   # <-- change this
SOURCE_SUBJECT = "A01T"                        # pretrained model subject
TARGET_SUBJECT = "A02T"                        # fine-tuning subject
MODEL_PATH = os.path.join(DATA_PATH, "models", f"{SOURCE_SUBJECT}_EEGNet")  # base name
SAVE_PATH = os.path.join(DATA_PATH, "transfer_results")
os.makedirs(SAVE_PATH, exist_ok=True)

FREEZE_LAYERS = 6   # how many layers to freeze
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-4

# -----------------------------
# STEP 1: LOAD MODEL (Handles .keras / .h5)
# -----------------------------
def load_eegnet_model(model_path):
    """Load model safely for both Keras 3 (.keras) and legacy .h5 formats."""
    if os.path.exists(model_path + ".h5"):
        print(f"✅ Loading .h5 model: {model_path}.h5")
        return load_model(model_path + ".h5")
    elif os.path.exists(model_path + ".keras"):
        print(f"✅ Loading .keras model: {model_path}.keras")
        return load_model(model_path + ".keras", compile=False)
    elif os.path.isdir(model_path):
        print(f"⚠️ Loading SavedModel folder using TFSMLayer (inference only)...")
        return TFSMLayer(model_path, call_endpoint="serving_default")
    else:
        raise FileNotFoundError(f"No model found at {model_path} (.h5 or .keras expected).")

model = load_eegnet_model(MODEL_PATH)

# -----------------------------
# STEP 2: PREPARE FOR FINE-TUNING
# -----------------------------
if hasattr(model, "layers"):
    print(f"Freezing first {FREEZE_LAYERS} layers...")
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=LR),
              metrics=["accuracy"])

# -----------------------------
# STEP 3: LOAD TARGET SUBJECT DATA
# -----------------------------
print(f"\nLoading target subject data: {TARGET_SUBJECT}")
X_train = np.load(os.path.join(DATA_PATH, f"{TARGET_SUBJECT}_X_train.npy"))
y_train = np.load(os.path.join(DATA_PATH, f"{TARGET_SUBJECT}_y_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, f"{TARGET_SUBJECT}_X_test.npy"))
y_test  = np.load(os.path.join(DATA_PATH, f"{TARGET_SUBJECT}_y_test.npy"))

# Reshape and encode
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
y_train = y_train - y_train.min()
y_test = y_test - y_test.min()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"Data ready: X_train={X_train.shape}, y_train={y_train.shape}")

# -----------------------------
# STEP 4: FINE-TUNE MODEL
# -----------------------------
print("\n🚀 Starting fine-tuning...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# -----------------------------
# STEP 5: EVALUATE & SAVE
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Fine-tuned Accuracy on {TARGET_SUBJECT}: {acc * 100:.2f}%")

# Save fine-tuned model and training history
fine_tuned_path_h5 = os.path.join(SAVE_PATH, f"{SOURCE_SUBJECT}_to_{TARGET_SUBJECT}_finetuned.h5")
fine_tuned_path_keras = os.path.join(SAVE_PATH, f"{SOURCE_SUBJECT}_to_{TARGET_SUBJECT}_finetuned.keras")
history_path = os.path.join(SAVE_PATH, f"{SOURCE_SUBJECT}_to_{TARGET_SUBJECT}_history.npy")

print("\n💾 Saving fine-tuned model in both formats...")
model.save(fine_tuned_path_h5, save_format="h5")
model.save(fine_tuned_path_keras)
np.save(history_path, history.history)

print(f"\n✅ Files saved in: {SAVE_PATH}")
print(f" - Model (.h5): {fine_tuned_path_h5}")
print(f" - Model (.keras): {fine_tuned_path_keras}")
print(f" - Training history: {history_path}")