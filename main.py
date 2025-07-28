"""
Fixed Forest Fire Prediction System
-----------------------------------
Improvements:
- Classification (fire vs. no fire) instead of regression for better handling of skewed data.
- Simpler LSTM with tuning.
- Sequences from sorted data (by month/day) to approximate time-series.
- Evaluation with classification metrics and graphs.

Requirements: tensorflow, scikit-learn, matplotlib, pandas, numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks #type: ignore

SEED = 42
SEQ_LEN = 30
TEST_RATIO = 0.20
BATCH_SIZE = 16  # Smaller for better generalization
EPOCHS = 100
PATIENCE = 10

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
df = pd.read_csv("forestfires.csv")

# Preprocessing
for col in ["month", "day"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Sort by month and day to approximate time-series
df = df.sort_values(by=['month', 'day']).reset_index(drop=True)

# Binary target: 1 if area > 0 (fire), 0 otherwise
df['fire'] = (df['area'] > 0).astype(int)
feature_cols = df.columns.drop(['area', 'fire'])
X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df['fire'].values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Create sequences
def make_windows(data, targets, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.asarray(X), np.asarray(y)

X_seq, y_seq = make_windows(X_scaled, y_raw, SEQ_LEN)

# Train/test split (chronological)
split_idx = int((1 - TEST_RATIO) * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# Calculate class weights for imbalanced dataset
neg, pos = np.bincount(y_train.astype(int))
total = neg + pos
print(f"\nTotal examples: {total}\nPositive examples: {pos} ({100 * pos / total:.2f}% of total)\n")

# Scaling by total to avoid very small loss value, then divide by number of examples of given class
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weight}")

# Model: Simpler LSTM for classification
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, X_train.shape[-1])),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(16),
    layers.Dense(8, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # Binary output
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Train with early stopping
cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_early],
    class_weight=class_weight, # Added class weight
    verbose=2
)

# Evaluate
y_pred_prob = model.predict(X_test).squeeze()
y_pred = (y_pred_prob > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot predicted vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Fire (1=Yes, 0=No)", marker='o')
plt.plot(y_pred, label="Predicted", marker='x', linestyle='--')
plt.xlabel("Test Samples")
plt.ylabel("Fire Occurrence")
plt.title("Predicted vs Actual Fire Occurrences")
plt.legend()
plt.show()

# Future prediction (using last 30 records)
latest_window = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
forecast_prob = model.predict(latest_window)[0, 0]
forecast = "HIGH RISK (Fire likely)" if forecast_prob > 0.5 else "LOW RISK (No fire expected)"
print(f"\nFuture Prediction: {forecast} (Probability: {forecast_prob:.2f})")
