# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:11:42 2024

@author: Nicolae 
"""
#%%Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#%% Enable GPU support
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Group the data by input_file
    grouped = data.groupby('input_file')
    
    X_list = []
    y_list = []
    
    for _, group in grouped:
        # Extract phase and amplitude values (assuming 5 frequencies)
        phases = group['phase'].values
        amplitudes = group['amplitude'].values
        
        # Combine phase and amplitude into a single input vector
        X = np.concatenate([phases, amplitudes])
        
        # Extract mua and musp (assuming they're constant for each input_file)
        y = group[['mua', 'musp']].iloc[0].values
        
        X_list.append(X)
        y_list.append(y)
    
    return np.array(X_list), np.array(y_list)

#%% Load the dataset

X, y = load_and_preprocess_data('dataset.csv')


#%% Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

#%% Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#%% Define the model architecture
model = Sequential([
    Dense(20, activation='tanh', input_shape=(12,)),
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(2)
])

#%% Define the optimizer
optimizer = Adam(learning_rate=0.001)

#%% Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

#%% Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

#%% Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=1000,
    batch_size=100,
    callbacks=[early_stopping],
    verbose=1
)


#%% EVALUATION 
#Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

#%% Make predictions on the test set
y_pred = model.predict(X_test_scaled)

#%% Calculate additional metrics
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
mape_scores = mean_absolute_percentage_error(y_test, y_pred, multioutput='raw_values')
rmse_scores = np.sqrt(np.mean((y_test - y_pred)**2, axis=0))

print("R-squared scores:")
for i, score in enumerate(['mua', 'musp']):
    print(f"{score}: {r2_scores[i]:.4f}")

print("\nMean Absolute Percentage Error:")
for i, score in enumerate(['mua', 'musp']):
    print(f"{score}: {mape_scores[i]:.4f}")

print("\nRoot Mean Squared Error:")
for i, score in enumerate(['mua', 'musp']):
    print(f"{score}: {rmse_scores[i]:.4f}")

#%% Learning Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

#%% Residual Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
output_names = ['mua', 'musp']

for i, ax in enumerate(axes):
    residuals = y_test[:, i] - y_pred[:, i]
    ax.scatter(y_pred[:, i], residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel(f'Predicted {output_names[i]}')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{output_names[i]} Residual Plot')

plt.tight_layout()
plt.show()

#%% K-Fold Cross-Validation
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)
    
    model = Sequential([
        Dense(20, activation='tanh', input_shape=(12,)),
        Dense(20, activation='tanh'),
        Dense(20, activation='tanh'),
        Dense(20, activation='tanh'),
        Dense(20, activation='tanh'),
        Dense(20, activation='tanh'),
        Dense(2)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    
    history = model.fit(
        X_train_fold_scaled, y_train_fold,
        validation_data=(X_val_fold_scaled, y_val_fold),
        epochs=100,
        batch_size=100,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
        verbose=0
    )
    
    val_loss, val_mae = model.evaluate(X_val_fold_scaled, y_val_fold, verbose=0)
    cv_scores.append(val_mae)
    print(f"Fold {fold+1} - Validation MAE: {val_mae:.4f}")

print(f"\nMean CV MAE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

#%% Feature Importance (using permutation importance)
"""
def permutation_importance(model, X, y, n_repeats=10):
    baseline_mae = np.mean(np.abs(y - model.predict(X)), axis=0)
    importances = np.zeros((X.shape[1], y.shape[1]))
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        for _ in range(n_repeats):
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_mae = np.mean(np.abs(y - model.predict(X_permuted)), axis=0)
            importances[i] += (permuted_mae - baseline_mae) / baseline_mae
    
    return importances / n_repeats

feature_importance = permutation_importance(model, X_test_scaled, y_test)

plt.figure(figsize=(10, 6))
plt.bar(range(10), feature_importance[:, 0], label='mua')
plt.bar(range(10), feature_importance[:, 1], bottom=feature_importance[:, 0], label='musp')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.legend()
plt.tight_layout()
plt.show()

"""