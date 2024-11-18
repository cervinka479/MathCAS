import pandas as pd

# Load the dataset
df = pd.read_csv('deleteme\dataset3D_10M_training_nstep180.csv')

# Columns to analyze
columns = ['ResVort', 'ResStrain', 'Shear']

# Initialize a dictionary to store the counts, percentages, and minimum values
counts = {col: {'zero': 0, 'non_zero': 0, 'zero_pct': 0.0, 'non_zero_pct': 0.0, 'min_value': None} for col in columns}

# Calculate the counts, percentages, and minimum values
for col in columns:
    total = len(df[col])
    counts[col]['zero'] = (df[col] == 0).sum()
    counts[col]['non_zero'] = (df[col] != 0).sum()
    counts[col]['zero_pct'] = (counts[col]['zero'] / total) * 100
    counts[col]['non_zero_pct'] = (counts[col]['non_zero'] / total) * 100
    counts[col]['min_value'] = df[col].min()

# Print the results
for col in columns:
    print(f"{col}: Zero values = {counts[col]['zero']} ({counts[col]['zero_pct']:.2f}%), Non-zero values = {counts[col]['non_zero']} ({counts[col]['non_zero_pct']:.2f}%), Minimum value = {counts[col]['min_value']}")