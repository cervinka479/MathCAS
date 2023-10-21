import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'ANN_DataSet\.filtered_dataset_28K.csv'

# Read the CSV file
original_df = pd.read_csv(file_path)
'''print(original_df)
print("divider")'''

mask = (original_df['X'] >= 5.0) & (original_df['X'] <= 5.5)
#mask = mask & (original_df['Y'] ** 2 + original_df['Z'] ** 2 <= 4)
mask = mask & ((original_df['A11']**2 + original_df['A21']**2 + original_df['A31']**2 + original_df['A12']**2 + original_df['A22']**2 + original_df['A32']**2 + original_df['A13']**2 + original_df['A23']**2 + original_df['A33']**2)>=8e-03)
#mask = mask & (original_df['ResVort'] == 0)

#print((original_df['A11']**2 + original_df['A21']**2 + original_df['A31']**2 + original_df['A12']**2 + original_df['A22']**2 + original_df['A32']**2 + original_df['A13']**2 + original_df['A23']**2 + original_df['A33']**2))

filtered_data = original_df[mask]

#print(filtered_data)

#(filtered_data['A11']**2 + filtered_data['A21']**2 + filtered_data['A31']**2 + filtered_data['A12']**2 + filtered_data['A22']**2 + filtered_data['A32']**2 + filtered_data['A13']**2 + filtered_data['A23']**2 + filtered_data['A33']**2).to_csv("filtered_dataset.csv", index=False, float_format='%.13E')

# Export the filtered dataset to a new CSV file
filtered_data.to_csv("filtered_dataset.csv", index=False, float_format='%.13E')