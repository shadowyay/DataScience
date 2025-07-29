import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('last1.csv')

# Remove leading/trailing spaces from column names to avoid KeyError
data.columns = data.columns.str.strip()

# Define the cleaning function for the 'Classes' column
def clean_classes(x):
    x = str(x).strip()  # Ensure x is a string and strip any extra spaces
    if x == "not fire":
        return 0
    elif x == "fire":
        return 1
    else:
        return np.nan  # Handle unexpected values (will be dropped later)

# Apply the cleaning function to the 'Classes' column
data['Classes'] = data['Classes'].apply(clean_classes)

# Drop the "year" column (assumed to be a constant value)
data = data.drop(columns=['year'])

# One-hot encode the "month" column
data = pd.get_dummies(data, columns=['month'], prefix='month')

# Drop rows with missing values (e.g., from unexpected 'Classes' values)
data = data.dropna()

# Split into features (X) and target (y)
X = data.drop(columns=['Classes'])
y = data['Classes']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Accuracy: {accuracy:.4f}')

# Predict and print detailed metrics
y_pred = (model.predict(X_test) > 0.5).astype(int)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Not Fire', 'Fire']))

# Plot training and validation metrics
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()