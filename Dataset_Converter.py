# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:04:20 2024

@author: Nicolae

"""

#%% Load used packages
import os
import sys
import csv
#%%
def extract_info(file_path, file_index):
    info = {
        'input_file': file_index,
        'mua': None,
        'musp': None,
        'n_int': None,
        'f_values': [],
        'amplitude_values': [],
        'phase_values': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('$mua,'):
                info['mua'] = line.split(',')[1].strip()
            elif line.startswith('$musp,'):
                info['musp'] = line.split(',')[1].strip()
            elif line.startswith('$n_int,'):
                info['n_int'] = line.split(',')[1].strip()
            elif not line.startswith('#') and not line.startswith('$'):
                values = line.strip().split(',')
                if len(values) == 3:
                    info['f_values'].append(values[0])
                    info['amplitude_values'].append(values[1])
                    info['phase_values'].append(values[2])
    
    return info
#%%
def combine_files(input_folder):
    output_data = []
    file_index = 0
    
    for filename in os.listdir(input_folder):
        if filename.startswith('test_result_') and filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            info = extract_info(file_path, file_index)
            
            for f, amplitude, phase in zip(info['f_values'], info['amplitude_values'], info['phase_values']):
                output_data.append([
                    info['input_file'],
                    info['mua'],
                    info['musp'],
                    info['n_int'],
                    f,
                    phase,
                    amplitude
                ])
            
            file_index += 1
    
    return output_data
#%%
def save_output(output_data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['input_file', 'mua', 'musp', 'n_int', 'f', 'phase', 'amplitude'])
        writer.writerows(output_data)
#%%
def main():
    input_folder = input("Please enter the path to the input folder: ")
    
    if not os.path.isdir(input_folder):
        print(f"Error: The directory '{input_folder}' does not exist.")
        sys.exit(1)
    
    output_data = combine_files(input_folder)
    output_file = os.path.join(input_folder, 'Output.csv')
    save_output(output_data, output_file)
    
    print(f"Output file has been saved as '{output_file}' in the input folder")
if __name__ == "__main__":
    main()