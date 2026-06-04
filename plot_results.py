"""
EEGNet Transfer Learning Results Visualization
Author: Safid Hasan
Description:
    Loads multiple training history .npy files and plots:
    - Training vs Validation Accuracy
    - Fine-tuned vs Baseline Comparison
    Generates IEEE-paper-ready figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = r"C:\Users\safid\Downloads\EEGNet\preprocessed_data\transfer_results"  # update this
histories = [
    ("Baseline A01T", os.path.join(DATA_PATH, "A01T_training_history.npy")),
    ("Fine-tuned A01T->A02T", os.path.join(DATA_PATH, "A01T_to_A02T_history.npy")),
    # Add more subjects/fine-tuning histories here
]

# -----------------------------
# STEP 1: LOAD HISTORIES
# -----------------------------
history_dict = {}
for name, path in histories:
    if os.path.exists(path):
        history_dict[name] = np.load(path, allow_pickle=True).item()
    else:
        print(f"⚠️ History file not found: {path}")

# -----------------------------
# STEP 2: PLOT ACCURACY COMPARISON
# -----------------------------
plt.figure(figsize=(10, 6))
for name, hist in history_dict.items():
    plt.plot(hist['accuracy'], label=f"{name} (train)")
    plt.plot(hist['val_accuracy'], '--', label=f"{name} (val)")

plt.title("EEGNet Training & Fine-Tuning Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "accuracy_comparison.png"), dpi=300)
plt.show()

# -----------------------------
# STEP 3: PLOT LOSS COMPARISON
# -----------------------------
plt.figure(figsize=(10, 6))
for name, hist in history_dict.items():
    plt.plot(hist['loss'], label=f"{name} (train)")
    plt.plot(hist['val_loss'], '--', label=f"{name} (val)")

plt.title("EEGNet Training & Fine-Tuning Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "loss_comparison.png"), dpi=300)
plt.show()