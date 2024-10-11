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
#%%Remove and Mask
def remove_and_mask_data(X, y, removal_percentage=0.0):
    mask = np.random.choice([True, False], size=X.shape, p=[1-removal_percentage, removal_percentage])
    X_masked = np.where(mask, X, np.nan)
    return X_masked, y
#%%preprocess and define noise levels 
def preprocess_data_with_augmentation(file_path, phase_noise_mean=0, phase_noise_var=0.00, 
                                      amplitude_noise_mean=0, amplitude_noise_var=0.00, 
                                      removal_percentage=0.0):
    X, y = load_and_preprocess_data(file_path)
    X_noisy, y_noisy = add_noise_to_dataset(X, y, phase_noise_mean, phase_noise_var, 
                                            amplitude_noise_mean, amplitude_noise_var)
    X_masked, y_masked = remove_and_mask_data(X_noisy, y_noisy, removal_percentage)
    return X_masked, y_masked

#%% 3d scatterplot
def create_3d_scatter_plot(X, y):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y[:, 0], cmap='viridis', s=1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.colorbar(scatter, label='mua')
    plt.title('3D Scatter Plot of First 3 Features')
    plt.show()
#%% interactive 3d scatterplot
def create_interactive_3d_plot(X, y):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=y[:, 0],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title='Interactive 3D Plot of First 3 Features'
    )
    # Generate filename with current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interactive_3d_plot_{current_time}.html"
    
    fig.write_html(filename)
    print(f"Interactive 3D plot saved to '{filename}'.")
#%% umap
def perform_umap_visualization(X, y):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y[:, 0], cmap='viridis', s=1)
    plt.colorbar(label='mua')
    plt.title('UMAP Visualization')
    plt.xlabel('Vector 1')
    plt.ylabel('Vector 2')
    plt.show()

#####-------------------------------MAIN CODE STARTS----------------------#######

#%% Load and preprocess the dataset with augmentation
X, y = preprocess_data_with_augmentation('dataset.csv',                                        #considering a 95% interval of +-3% noise lovel: 
                                         phase_noise_mean= 0.000, phase_noise_var=0.000,   #phase_noise_mean= -0.0093, phase_noise_var=0.00385,
                                         amplitude_noise_mean=0, amplitude_noise_var=0.000,  #amplitude_noise_mean=0, amplitude_noise_var=0.00893
                                         removal_percentage=0.0)
#%% analyzing datatset
create_interactive_3d_plot(X, y)


#%% Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=137)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

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
    epochs=300,
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


#%% Learning Curves
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
    ax.set_title(f' Residual Plot for {output_names[i]} 3x tanh')

plt.tight_layout()
plt.show()

#%% K-Fold Cross-Validation
k_fold = KFold(n_splits=10, shuffle=True, random_state=137)
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


#%% Save the model
model.save('DNN_24x6_no-noise.keras')

#%% Load the model (for demonstration purposes)
loaded_model = load_model('DNN_24x6_no-noise.keras')


#%% Export the model to TensorFlow Lite format

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

#%% Save the TensorFlow Lite model
with open('DNN_24x6_no-noise.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model exported to TensorFlow Lite format: .tflite")


#%% Define the model architecture with masking
'''
input_layer = Input(shape=(12,))
masked_input = Masking(mask_value=np.nan)(input_layer)
x = Dense(24, activation='tanh')(masked_input)
for _ in range(2):
    x = Dense(24, activation='tanh')(x)
output_layer = Dense(2)(x)

model = Model(inputs=input_layer, outputs=output_layer) '''