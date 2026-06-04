"""
EEGNet Training Script for BCI Competition IV-2a (Updated for Keras 3)
Author: Safid Hasan
Description:
    Loads preprocessed EEG .npy files, trains EEGNet, and saves both .h5 and .keras formats.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, DepthwiseConv2D,
                                     SeparableConv2D, AveragePooling2D, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.utils import to_categorical

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = r"C:\Users\safid\Downloads\EEGNet\preprocessed_data"   # <-- change this
SUBJECT = "A01T"                               # <-- match your subject file
EPOCHS = 200
BATCH_SIZE = 16
SAVE_PATH = os.path.join(DATA_PATH, "models")
os.makedirs(SAVE_PATH, exist_ok=True)

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
print(f"Loading data for {SUBJECT}...")
X_train = np.load(os.path.join(DATA_PATH, f"{SUBJECT}_X_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, f"{SUBJECT}_X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, f"{SUBJECT}_y_train.npy"))
y_test  = np.load(os.path.join(DATA_PATH, f"{SUBJECT}_y_test.npy"))

# -----------------------------
# STEP 2: PREPARE DATA FOR EEGNet
# -----------------------------
X_train = X_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

y_train = y_train - y_train.min()
y_test = y_test - y_test.min()

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

n_channels = X_train.shape[1]
n_samples  = X_train.shape[2]
n_classes  = y_train_cat.shape[1]

print(f"Data shape: {X_train.shape}, Classes: {n_classes}")

# -----------------------------
# STEP 3: DEFINE EEGNet MODEL
# -----------------------------
model = Sequential([
    Conv2D(8, (1, 64), padding='same', input_shape=(n_channels, n_samples, 1), use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2),
    BatchNormalization(),
    AveragePooling2D((1, 4)),
    Dropout(0.25),
    SeparableConv2D(16, (1, 16), use_bias=False, padding='same'),
    BatchNormalization(),
    AveragePooling2D((1, 8)),
    Dropout(0.25),
    Flatten(),
    Dense(n_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# STEP 4: TRAIN MODEL
# -----------------------------
print("\nStarting training...\n")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# -----------------------------
# STEP 5: EVALUATE & SAVE MODELS
# -----------------------------
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Save both formats
h5_path = os.path.join(SAVE_PATH, f"{SUBJECT}_EEGNet.h5")
keras_path = os.path.join(SAVE_PATH, f"{SUBJECT}_EEGNet.keras")

print("\nSaving model in both formats...")
model.save(h5_path, save_format='h5')
model.save(keras_path)  # New Keras 3 format
np.save(os.path.join(SAVE_PATH, f"{SUBJECT}_training_history.npy"), history.history)

print(f"Models saved:\n - {h5_path}\n - {keras_path}")
print(f"Training history saved to: {SAVE_PATH}")
