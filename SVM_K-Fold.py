# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:59:48 2025

@author: J. Saldaña
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import the SVC (Support Vector Classifier) algorithm
from sklearn.svm import SVC

# --- Custom Information Prints ---
print("ID PX: 0001")
print("Test: ID Test: Cambios de velocidad")
# -----------------------------------

# 1. Load the dataset
df = pd.read_csv('Dataset_Marcha_acelerometroPX.csv')

# 2. Select relevant columns
X = df[['x', 'y', 'z']]  # Independent variables (features)
y = df['Clase']        # Target variable (class)

# 3. Scale the data
# Scaling is crucial for SVM performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Convert data into time sequences (timesteps)
# This function converts your time series data into sequences.
def create_sequences(X, y, timesteps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps + 1):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps-1])  # Label associated with the last timestep of the sequence
    return np.array(X_seq), np.array(y_seq)

timesteps = 10
X_seq_original, y_seq_original = create_sequences(X, y, timesteps)

# --- Flatten sequences for the SVC algorithm ---
# Traditional ML algorithms like SVC expect a 2D input (samples, flattened_features).
# We reshape X_seq_original from (samples, timesteps, num_features) to (samples, timesteps * num_features).
X_seq_flattened = X_seq_original.reshape(X_seq_original.shape[0], -1)

# 5. Define the SVC model
# 'random_state' ensures reproducibility. The kernel is set to 'sigmoid' as per your provided file.
# Note: SVM can be computationally intensive, especially with non-linear kernels on large datasets.
svc_model = SVC(kernel='sigmoid', random_state=42)

# 6. Implement K-Fold Cross-Validation
n_splits = 5 # Define the number of folds (e.g., 5 or 10 is common)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # 'shuffle=True' to shuffle the data

print(f"\n======== Running K-Fold Cross-Validation for: SVC (Sigmoid Kernel) ========")

all_y_test = []
all_y_pred = []
all_accuracies = []

fold = 1
# Iterate through the folds using the flattened data for SVC
for train_index, test_index in kf.split(X_seq_flattened, y_seq_original):
    print(f"--- Fold {fold}/{n_splits} ---")
    X_train, X_test = X_seq_flattened[train_index], X_seq_flattened[test_index]
    y_train, y_test = y_seq_original[train_index], y_seq_original[test_index]

    # Create a new instance of the model for each fold
    # This ensures that the training of each fold is independent.
    model_instance = SVC(kernel='sigmoid', random_state=42)

    # Train the model
    model_instance.fit(X_train, y_train)

    # Evaluate the model
    y_pred_fold = model_instance.predict(X_test)
    fold_accuracy = accuracy_score(y_test, y_pred_fold)

    print(f"Accuracy for Fold {fold}: {fold_accuracy:.4f}")
    print(f"\nClassification Report for Fold {fold}:")
    print(classification_report(y_test, y_pred_fold))

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred_fold)
    all_accuracies.append(fold_accuracy)

    fold += 1

print(f"\n--- Overall Results for SVM ---")
print("Average Accuracy across folds:", np.mean(all_accuracies))
print("Overall Accuracy (from concatenated predictions across all folds):", accuracy_score(all_y_test, all_y_pred))
print("Overall Classification Report (from concatenated predictions):")
print(classification_report(all_y_test, all_y_pred))

# 7. Display the Confusion Matrix for the complete set of predictions
cm = confusion_matrix(all_y_test, all_y_pred)
# Use unique labels from y_seq_original to ensure correct display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_seq_original))
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Overall Confusion Matrix - SVM (K-Fold Cross-Validation)")
plt.show()

print("\nSVC (Sigmoid Kernel) model evaluation completed.")