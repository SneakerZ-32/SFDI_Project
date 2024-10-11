# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:47:00 2024

@author: Nicolae
"""
#####-------------------------------IMPORTS----------------------#######
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import umap
import plotly.graph_objects as go
from scipy.stats import pearsonr
from datetime import datetime
#####-------------------------------DEFINING FUNCTIONS----------------------#######
#%% Enable GPU support
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
#%% Check for GPU
gpu_available = tf.test.is_gpu_available()

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
    
#%% Just load a  dataset

X, y = load_and_preprocess_data('OutputDataset-of-Generate-Noisy-Dataset.-py-.csv')

#%% Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=137) #splitting into training and validation+test
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=137) #splitting validation & test

#%% Standardize the input features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#%% Defined the current best model architecture

model = Sequential([
    Masking(mask_value=np.nan, input_shape=(12,)),
    Dense(24, activation='silu'),
    Dense(24, activation='mish'),
    Dense(24, activation='mish'),
    Dense(24, activation='silu'),
    Dense(2)
])

#%% Define the optimizer( BEST OPTIMIZER YET!!!) only 100 epochs for 10^-4 MAE
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.99,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=0.0001,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="adam",
    )

    

#%% Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

#%% Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)

#%% Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=1000,
    batch_size=1024,
    callbacks=[early_stopping],
    verbose=1
)


#%% Evaluate the loaded model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

#%% Make predictions using the loaded model
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


#%% Learning Curves with log y axis
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.yscale("log")
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.yscale("log")
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
    ax.set_title(f' Residual Plot for {output_names[i]}')

plt.tight_layout()
plt.show()


#%% RUN THE FOLLOWING CELLS ONLY IF YOU´RE INTERESTED IN K-FOlD CROSS VALIDATION 
#%% K-Fold cross validation v2

# Define the k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=137)

# Initialize arrays to store results
fold_mae_scores = []
fold_mse_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold+1}/{k}")

    # Split the data into training and validation sets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define the model architecture (re-initialize for each fold)
    model = Sequential([
        Masking(mask_value=np.nan, input_shape=(12,)),
        Dense(24, activation='silu'),
        Dense(24, activation='mish'),
        Dense(24, activation='mish'),
        Dense(24, activation='silu'),
        Dense(2)
    ])

    # Define the optimizer
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.99,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=0.0001,
    )

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)

    # Train the model on this fold's training data
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=200,
        batch_size=1024,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on this fold's validation data
    val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
    
    print(f"Fold {fold+1} MAE: {val_mae:.4f}")
    
    # Store the scores
    fold_mae_scores.append(val_mae)
    fold_mse_scores.append(val_loss)

# Calculate the average performance metrics across all folds
average_mae = np.mean(fold_mae_scores)
average_mse = np.mean(fold_mse_scores)
print(f"\nAverage MAE across all folds: {average_mae:.4f}")
print(f"Average MSE across all folds: {average_mse:.4f}")
#%% define model name 
model_name = 'silu_x2mish_silu_200epochs_no-noise_retrainedon1p5_no-mask.keras'
#%%#%% Save the model
model.save(model_name)

#%% Load the model (for demonstration purposes)
loaded_model = load_model(model_name)
#%% Export the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

#%% Save the TensorFlow Lite model
tflite_model_name = 'silu_x2mish_silu300epochs_no_noise_no_mask.tflite'
with open(tflite_model_name, 'wb') as f:
    f.write(tflite_model)

print("Model exported to TensorFlow Lite format: .tflite")

