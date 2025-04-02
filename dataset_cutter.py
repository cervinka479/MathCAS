import pandas as pd

# Define the column names for the 9 elements of matrix A
columns = ['A11', 'A12', 'A13', 'A21', 'A22', 'A23', 'A31', 'A32', 'A33', 'ResVort', "ResStrain", "Shear"]

# Read the required rows from the CSV file, selecting only the specified columns
df = pd.read_csv(r"deleteme\dataset_compressible_flow_60M_training_nstep180.csv", usecols=columns, skiprows=range(1, 59500000), nrows=500000)

# Write these selected columns to a new CSV file
df.to_csv(r"deleteme\dataset_compressible_flow_500K_test_nstep180.csv", index=False)