import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'temp_preprocessed.csv'

# Read the CSV file
original_df = pd.read_csv(file_path)
print(original_df)
print("divider")

mask = (original_df['X'] >= 5.0) & (original_df['X'] <= 5.5) #& (original_df['Y'] ** 2 + original_df['Z'] ** 2 <= 4)

filtered_data = original_df[mask]

print(filtered_data)

# Export the filtered dataset to a new CSV file
filtered_data.to_csv("filtered_dataset2.csv", index=False, float_format='%.13E')