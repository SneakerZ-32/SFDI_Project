# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:24:47 2024

@author: Nicolaew
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
#%%
def sigma_calculator(mean = 1,lower_bound = 1,upper_bound = 1,confidence_level = 0.99):
    # Given values
    mean = 1
    lower_bound = 1
    lower_bound = 1
    confidence_level = 0.99
    
    # z-score for 99% confidence level (two-tailed)
    z_score = norm.ppf((1 + confidence_level) / 2)
    
    # Calculate sigma
    sigma = (upper_bound - mean) / z_score
    return sigma
#%% Multiply by a value around 1 to add noise
def add_noise(value):
    # Generate a noise factor from a normal distribution
    # with mean 1 and standard deviation 
    sigma = sigma_calculator(mean = 1, lower_bound =0.91,upper_bound = 1.09,confidence_level = 0.99)# Define HERE WHAT YOUR SETTING ARE
    noise_factor = np.random.normal(loc=1, scale= sigma)
    
    # Clip the noise factor to ensure it's between 0.97 and 1.03
    #noise_factor = np.clip(noise_factor, 0.97, 1.03)
    
    # Multiply the original value by the noise factor
    return value * noise_factor

#%% combine functions
def process_dataset(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Apply noise to 'phase' and 'amplitude' columns
    df['phase'] = df['phase'].apply(add_noise)
    df['amplitude'] = df['amplitude'].apply(add_noise)
    
    # Write the result to a new CSV file
    df.to_csv(output_file, index=False)
    
    return df
#%%Generate Noisy Dataset
# Example usage
input_file = 'dataset.csv'
output_file = 'output_dataset_with_noise_9percent.csv'

result_df = process_dataset(input_file, output_file)


print(f"\nProcessed dataset has been saved to {output_file}")