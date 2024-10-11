# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:14:54 2024

@author: Nicolai
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import umap
from datetime import datetime

#%% Load the dataset
df = pd.read_csv('output_dataset_with_noise_0percent.csv')
df['phase'] = df['phase'].abs() #phase is negative and we cannot do log of negative numbers

#%% Allow the user to see the available ranges
print("Current data range:")
print(f"input_file range: {df['input_file'].min()} to {df['input_file'].max()}")
print(f"f range: {df['f'].min()} to {df['f'].max()}")
print(f"mua range: {df['mua'].min()} to {df['mua'].max()}")
print(f"musp range: {df['musp'].min()} to {df['musp'].max()}")
print(f"phase range: {df['phase'].min()} to {df['phase'].max()}")
print(f"amplitude range: {df['amplitude'].min()} to {df['amplitude'].max()}")
print(f"n_int range: {df['n_int'].min()} to {df['n_int'].max()}")



#%% Allow the user to define a subset of the data to analyze
#subset = df[(df['input_file'] >= 0) & (df['input_file'] <= 90159)] #We basically select the whole datatset, it doesn´t take that much to plot
subset = df
#%%Create an interactive 3D plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=subset['amplitude'],
    y=subset['phase'],
    z=subset['mua'],
    mode='markers',
    marker=dict(
        size=2,
        color=subset['f'],
        colorscale='Viridis',
        opacity=1
    )
)])


fig.update_layout(
    scene=dict(
        xaxis_title='amplitude',
        yaxis_title='phase',
        zaxis_title='mua',
        xaxis_type='log',
        yaxis_type='log',
        zaxis_type='log'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Generate filename with current timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"interactive_3d_plot_{current_time}.html"

fig.write_html(filename)
print(f"Interactive 3D plot saved to '{filename}'.")

#%% UMAP Analysis
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

'''
#%% OLD Function to add noise to phase and amplitude
def add_noise_to_dataframe(df, phase_noise_mean=0, phase_noise_var=0.00, amplitude_noise_mean=0, amplitude_noise_var=0.000):
    # Create a copy of the DataFrame to avoid modifying the original
    noisy_df = df.copy()
    
    # Add noise to the 'phase' column
    noisy_df['phase'] += np.random.normal(phase_noise_mean, np.sqrt(phase_noise_var), df['phase'].shape)
    
    # Add noise to the 'amplitude' column
    noisy_df['amplitude'] += np.random.normal(amplitude_noise_mean, np.sqrt(amplitude_noise_var), df['amplitude'].shape)

    return noisy_df


noisy_df = add_noise_to_dataframe(df, phase_noise_mean= 0.004965,        #for 99% CI of +-3% noise phase_noise_mean= 0.0093     //  for 99% CI of +-1.5% noise phase_noise_mean= 0.004965
                                         phase_noise_var=0.00192,        #for 99% CI of +-3% noise phase_noise_var=0.00385       //  for 99% CI of +-1.5% noise noise phase_noise_var=0.00192
                                          amplitude_noise_mean=0,        #for 99% CI of +-3% noise amplitude_noise_mean=0        //  for 99% CI of +-1.5% noise amplitude_noise_mean=0
                                         amplitude_noise_var=0.0045)     #for 99% CI of +-3% noise amplitude_noise_var=0.00893   //  for 99% CI of +-1.5% noise amplitude_noise_var=0.00450


#%% Function to add noise to phase and amplitude( by multiplying by 1+-0.03%)
def add_noise_to_dataframe(df, phase_noise_mean=0, phase_noise_var=0.00, amplitude_noise_mean=0, amplitude_noise_var=0.000):
    # Create a copy of the DataFrame to avoid modifying the original
    noisy_df = df.copy()
    
    # Generate multiplicative noise factors for the 'phase' column
    phase_noise_factors = np.random.normal(phase_noise_mean, np.sqrt(phase_noise_var), df['phase'].shape)
    #phase_noise_factors = np.exp(phase_noise_factors)  # Ensure factors are positive

    # Apply multiplicative noise to the 'phase' column
    noisy_df['phase'] *= phase_noise_factors
    
    # Generate multiplicative noise factors for the 'amplitude' column
    amplitude_noise_factors = np.random.normal(amplitude_noise_mean, np.sqrt(amplitude_noise_var), df['amplitude'].shape)
    #amplitude_noise_factors = np.exp(amplitude_noise_factors)  # Ensure factors are positive
    
    # Apply multiplicative noise to the 'amplitude' column
    noisy_df['amplitude'] *= amplitude_noise_factors

    return noisy_df

noisy_df = add_noise_to_dataframe(df, phase_noise_mean= 1,        #for 99% CI of +-3% noise phase_noise_mean= 0.0093     //  for 99% CI of +-1.5% noise phase_noise_mean= 0.004965
                                         phase_noise_var=0.005823,        #for 99% CI of +-3% noise phase_noise_var=0.00385       //  for 99% CI of +-1.5% noise noise phase_noise_var=0.00192
                                          amplitude_noise_mean=1,        #for 99% CI of +-3% noise amplitude_noise_mean=0        //  for 99% CI of +-1.5% noise amplitude_noise_mean=0
                                         amplitude_noise_var=0.005823)     #for 99% CI of +-3% noise amplitude_noise_var=0.00893   //  for 99% CI of +-1.5% noise amplitude_noise_var=0.00450
#%%
noisy_df.to_csv('noisy_dataset1.csv', index=False)
'''