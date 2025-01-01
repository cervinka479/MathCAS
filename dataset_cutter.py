import pandas as pd

# Define the column names for the 9 elements of matrix A
columns = ['A11', 'A12', 'A13', 'A21', 'A22', 'A23', 'A31', 'A32', 'A33', 'ResVort', "ResStrain", "Shear"]

# Read the first lines from the CSV file, selecting only the specified columns
df = pd.read_csv(r"deleteme\dataset_compressible_flow_10M_training_nstep180.csv", usecols=columns, nrows=5000000)

# Write these selected columns to a new CSV file
df.to_csv(r"deleteme\dataset_compressible_flow_5M_training_nstep180.csv", index=False)