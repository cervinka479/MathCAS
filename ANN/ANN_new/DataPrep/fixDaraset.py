import pandas as pd

# Read the CSV file and preprocess to remove trailing commas
with open("sphere_Re300_training_nstep180.csv", 'r') as file:
    lines = file.readlines()
    lines = [line.rstrip(',\n') + '\n' for line in lines]

# Write the preprocessed lines to a new temporary file
with open("temp_preprocessed.csv", 'w') as temp_file:
    temp_file.writelines(lines)
