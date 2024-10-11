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
    
#%%
gpu_available = tf.test.is_gpu_available()
    
#%% Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Group the data by input_file
    grouped = data.groupby('input_file')
    
    X_list = []
    y_list = []
    
    for _, group in grouped:
        # Extract phase and amplitude values (assuming 6 frequencies)
        phases = group['phase'].values
        amplitudes = group['amplitude'].values
        
        # Combine phase and amplitude into a single input vector
        X = np.concatenate([phases, amplitudes])
        
        # Extract mua and musp (constant for each input_file)
        y = group[['mua', 'musp']].iloc[0].values
        
        X_list.append(X)
        y_list.append(y)
        
        
    
    return np.array(X_list), np.array(y_list)

#%%Add noise
def add_noise_to_dataset(X, y, phase_noise_mean=0, phase_noise_var=0.00, amplitude_noise_mean=0, amplitude_noise_var=0.000):
    noisy_X = X.copy()
    num_features = X.shape[1] // 2  # Assuming the first half are phases and the second half are amplitudes
    
    # Add noise to phases
    noisy_X[:, :num_features] += np.random.normal(phase_noise_mean, np.sqrt(phase_noise_var), (X.shape[0], num_features))
    
    # Add noise to amplitudes
    noisy_X[:, num_features:] += np.random.normal(amplitude_noise_mean, np.sqrt(amplitude_noise_var), (X.shape[0], num_features))
    
    return np.vstack((X, noisy_X)), np.vstack((y, y))

#%%Add noise v2
def add_noise_to_dataset(X, y, phase_noise_mean=0, phase_noise_var=0.00, amplitude_noise_mean=0, amplitude_noise_var=0.000):
    noisy_X = X.copy()
    num_features = X.shape[1] // 2  # Assuming the first half are phases and the second half are amplitudes
    
    # Generate multiplicative noise factors for phases
    phase_noise_factors = np.random.normal(phase_noise_mean, np.sqrt(phase_noise_var), (X.shape[0], num_features))
    #phase_noise_factors = np.exp(phase_noise_factors)  # Convert to multiplicative factors
    
    # Apply multiplicative noise to phases
    noisy_X[:, :num_features] *= phase_noise_factors
    
    # Generate multiplicative noise factors for amplitudes
    amplitude_noise_factors = np.random.normal(amplitude_noise_mean, np.sqrt(amplitude_noise_var), (X.shape[0], num_features))
    #amplitude_noise_factors = np.exp(amplitude_noise_factors)  # Convert to multiplicative factors
    
    # Apply multiplicative noise to amplitudes
    noisy_X[:, num_features:] *= amplitude_noise_factors
    
    return np.vstack((X, noisy_X)), np.vstack((y, y))

#%%Add Noise v3
def add_noise_to_dataset(X, y, phase_noise_mean=1, phase_noise_var=0.0, amplitude_noise_mean=1, amplitude_noise_var=0.0):
    noisy_X = X.copy()
    num_features = X.shape[1] // 2  # Assuming the first half are phases and the second half are amplitudes
    
    # Generate noise for phases and multiply
    phase_noise = np.random.normal(phase_noise_mean, np.sqrt(phase_noise_var), (X.shape[0], num_features))
    noisy_X[:, :num_features] *= phase_noise
    
    # Generate noise for amplitudes and multiply
    amplitude_noise = np.random.normal(amplitude_noise_mean, np.sqrt(amplitude_noise_var), (X.shape[0], num_features))
    noisy_X[:, num_features:] *= amplitude_noise
    
    return np.vstack((X, noisy_X)), np.vstack((y, y))





#%%Remove and Mask
def remove_and_mask_data(X, y, removal_percentage=0.0):
    mask = np.random.choice([True, False], size=X.shape,
                            p=[1-removal_percentage, removal_percentage])
    X_masked = np.where(mask, X, np.nan)
    return X_masked, y
#%%preprocess and define noise levels 
def preprocess_data_with_augmentation(file_path, phase_noise_mean=0, phase_noise_var=0, 
                                      amplitude_noise_mean=0, amplitude_noise_var=0, 
                                      removal_percentage=0.0):
    X, y = load_and_preprocess_data(file_path)
    X_noisy, y_noisy = add_noise_to_dataset(X, y, phase_noise_mean, phase_noise_var, 
                                            amplitude_noise_mean, amplitude_noise_var)
    X_masked, y_masked = remove_and_mask_data(X_noisy, y_noisy, removal_percentage)
    return X_masked, y_masked

#####-------------------------------MAIN CODE STARTS----------------------#######

#%% Load the dataset and add  noise
X, y = preprocess_data_with_augmentation('dataset.csv',                                        #considering a 95% interval of +-3% noise lovel: 
                                         phase_noise_mean= 1, phase_noise_var= 0,   #phase_noise_mean= -0.0093, phase_noise_var=0.00385,
                                         amplitude_noise_mean=1, amplitude_noise_var= 0,  #amplitude_noise_mean=0, amplitude_noise_var=0.00893
                                         removal_percentage=0.0)
   
#%% Just load a  dataset

X, y = load_and_preprocess_data('output_dataset_with_noise_0percent.csv')

#%% Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=137) #splitting into training and validation+test
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=137) #splitting validation & test

#%% Logarithmic transformation & Standardize the input features
'''
X_train_log = np.log1p(X_train) #it´s not a good idea because we have negative values and you can´t tranform negative values, the scaler alone is enough
X_val_log = np.log1p(X_val)
X_test_log = np.log1p(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)
X_test_scaled = scaler.transform(X_test_log)
'''
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

#%% K-Fold Cross-Validation
'''
k_fold = KFold(n_splits=5, shuffle=True, random_state=137)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)
    
    model = Sequential([
       Masking(mask_value=np.nan, input_shape=(12,)),
       Dense(24, activation='silu'),
       Dense(24, activation='mish'),
       Dense(24, activation='mish'),
       Dense(24, activation='silu'),
       Dense(2)
   ])

    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    history = model.fit(
        X_train_fold_scaled, y_train_fold,
        validation_data=(X_val_fold_scaled, y_val_fold),
        epochs=10,
        batch_size=100,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
        verbose=0
    )
    
    val_loss, val_mae = model.evaluate(X_val_fold_scaled, y_val_fold, verbose=0)
    cv_scores.append(val_mae)
    print(f"Fold {fold+1} - Validation MAE: {val_mae:.4f}")

print(f"\nMean CV MAE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
'''