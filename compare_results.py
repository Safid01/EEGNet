"""
compare_with_test_eval.py

Description:
    Loads training histories and saved models, evaluates each model on the
    corresponding subject's test set, and produces a comparison table with:
      - Final validation accuracy (last epoch)
      - Best validation accuracy (max val_accuracy)
      - Test accuracy (model evaluated on saved X_test / y_test)

Notes:
    - Models supported: .h5 and .keras (Keras3). If only a SavedModel folder exists
      and is not loadable for retraining/fine-tuning, evaluation will be skipped.
    - Expects preprocessed files like: A01T_X_test.npy, A01T_y_test.npy
    - Expects histories as: A01T_training_history.npy or A01T_to_A02T_history.npy
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -----------------------------
# CONFIGURATION - EDIT THESE
# -----------------------------
DATA_PATH = r"C:\Users\safid\Downloads\EEGNet\preprocessed_data"            # folder where subject npy files live
RESULTS_PATH = os.path.join(DATA_PATH, "transfer_results")  # folder where histories and models live
os.makedirs(RESULTS_PATH, exist_ok=True)

# List your experiments here:
# Each entry: (label, history_filename, model_basename, test_subject_prefix)
# - history_filename: .npy file saved earlier (history.history)
# - model_basename: path without extension; script will attempt .h5 then .keras then folder
# - test_subject_prefix: prefix string used to load X_test/y_test (e.g., "A02T")
experiments = [
    ("Baseline A01T", "A01T_training_history.npy", os.path.join(DATA_PATH, "models", "A01T_EEGNet"), "A01T"),
    ("Fine-tuned A01T→A02T", "A01T_to_A02T_history.npy", os.path.join(RESULTS_PATH, "A01T_to_A02T_finetuned"), "A02T"),
    ("Fine-tuned A01T→A03T", "A01T_to_A03T_history.npy", os.path.join(RESULTS_PATH, "A01T_to_A03T_finetuned"), "A03T"),
    # add more experiments as needed
]

# model compile settings (used if loaded model isn't compiled)
COMPILE_LOSS = "categorical_crossentropy"
COMPILE_OPT = Adam(learning_rate=1e-4)
COMPILE_METRICS = ["accuracy"]

# -----------------------------
# UTIL: safe model loader
# -----------------------------
def try_load_model(base_path):
    """
    Attempts to load a model given a base_path (without extension).
    Tries base_path + ".h5", then base_path + ".keras", then base_path (folder).
    Returns (model, available_for_eval_bool, loaded_path_str)
    """
    h5 = base_path + ".h5"
    keras_v3 = base_path + ".keras"
    folder = base_path  # might be a SavedModel folder

    # Try .h5 first
    if os.path.exists(h5):
        print(f"-> Loading .h5 model: {h5}")
        model = load_model(h5)
        return model, True, h5

    # Then .keras
    if os.path.exists(keras_v3):
        print(f"-> Loading .keras model: {keras_v3}")
        # load_model will load .keras format
        model = load_model(keras_v3)
        return model, True, keras_v3

    # Then try folder (SavedModel). Keras3 loader may or may not support full reload for training.
    if os.path.isdir(folder):
        try:
            print(f"-> Attempting to load folder model: {folder}")
            model = load_model(folder)
            return model, True, folder
        except Exception as e:
            # If can't load as trainable model, we cannot evaluate with `.evaluate` reliably.
            print(f"⚠️ Found SavedModel folder but could not load it as a trainable Keras model: {e}")
            return None, False, folder

    # Not found
    return None, False, base_path

# -----------------------------
# PROCESS EXPERIMENTS
# -----------------------------
rows = []
for label, history_fname, model_base, test_sub in experiments:
    hist_path = os.path.join(RESULTS_PATH, history_fname)
    history_exists = os.path.exists(hist_path)

    # load history metrics if available
    if history_exists:
        hist = np.load(hist_path, allow_pickle=True).item()
        # handle both possible naming ('val_accuracy' or 'val_acc') gracefully
        val_key_candidates = ["val_accuracy", "val_acc"]
        val_key = next((k for k in val_key_candidates if k in hist), None)

        if val_key is not None:
            final_val = hist[val_key][-1] * 100
            best_val = max(hist[val_key]) * 100
        else:
            final_val = None
            best_val = None
    else:
        final_val = None
        best_val = None
        print(f"⚠️ History missing for '{label}': {hist_path}")

    # Attempt to load the corresponding model
    model, evaluable, loaded_path = try_load_model(model_base)

    test_acc = None
    test_msg = ""
    if evaluable and model is not None:
        # try to load test data for test_sub
        X_test_path = os.path.join(DATA_PATH, f"{test_sub}_X_test.npy")
        y_test_path = os.path.join(DATA_PATH, f"{test_sub}_y_test.npy")
        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            try:
                X_test = np.load(X_test_path)
                y_test = np.load(y_test_path)

                # reshape for EEGNet if needed: (trials, channels, samples) -> add last axis
                if X_test.ndim == 3:
                    X_test_proc = X_test[..., np.newaxis]
                else:
                    X_test_proc = X_test

                # normalize labels to start at 0 and one-hot encode
                y_test_proc = y_test - y_test.min()
                y_test_cat = to_categorical(y_test_proc)

                # Ensure model is compiled; if not compiled, compile with default settings
                try:
                    # model.evaluate will raise if not compiled in some cases; safe to compile
                    if not hasattr(model, "optimizer") or model.optimizer is None:
                        model.compile(loss=COMPILE_LOSS, optimizer=COMPILE_OPT, metrics=COMPILE_METRICS)
                except Exception:
                    model.compile(loss=COMPILE_LOSS, optimizer=COMPILE_OPT, metrics=COMPILE_METRICS)

                loss, acc = model.evaluate(X_test_proc, y_test_cat, verbose=0)
                test_acc = acc * 100
                test_msg = f"Evaluated using {loaded_path}"
            except Exception as e:
                test_msg = f"Error during evaluation: {e}"
                print(f"⚠️ {label} - evaluation failed: {e}")
        else:
            test_msg = f"Missing test files for {test_sub}"
            print(f"⚠️ {label} - missing test files: {X_test_path} or {y_test_path}")
    else:
        test_msg = "Model not loadable for evaluation (.h5/.keras missing or folder not supported)"

    rows.append({
        "Model": label,
        "History File": history_fname if history_exists else "N/A",
        "Final Val Acc (%)": f"{final_val:.2f}" if final_val is not None else "N/A",
        "Best Val Acc (%)": f"{best_val:.2f}" if best_val is not None else "N/A",
        "Test Subject": test_sub,
        "Test Acc (%)": f"{test_acc:.2f}" if test_acc is not None else "N/A",
        "Note": test_msg
    })

# -----------------------------
# BUILD & SAVE TABLE
# -----------------------------
df = pd.DataFrame(rows, columns=[
    "Model", "History File", "Final Val Acc (%)", "Best Val Acc (%)",
    "Test Subject", "Test Acc (%)", "Note"
])

print("\n=== Comparison Table ===\n")
print(df.to_string(index=False))

# Save CSV and LaTeX
# Save CSV and LaTeX
csv_out = os.path.join(RESULTS_PATH, "comparison_table_with_test.csv")
latex_out = os.path.join(RESULTS_PATH, "comparison_table_with_test.tex")
df.to_csv(csv_out, index=False)

# Fix: use UTF-8 encoding to support arrows and symbols
with open(latex_out, "w", encoding="utf-8") as f:
    f.write(df.to_latex(index=False,
                        caption="EEGNet Transfer Learning Results (Validation & Test Accuracies)",
                        label="tab:comparison"))

print(f"\nSaved comparison CSV: {csv_out}")
print(f"Saved LaTeX table: {latex_out}")