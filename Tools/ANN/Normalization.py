import pandas as pd
import numpy as np

path = r"ResearchLog\2024\1_4\dOmegaRES100k.csv"

# Load the CSV dataset
data = pd.read_csv(path)

# Define the indices for inputs and outputs
i = [1, 3]  # Assuming 'Ux', 'Uy', 'Vx' are inputs
o = [4, 4]  # Assuming 'omegaRES' is output

# Extract the inputs and outputs
input_tensors = data.iloc[:, i[0]-1:i[1]].values
output_tensors = data.iloc[:, o[0]-1:o[1]].values

# Normalize the dataset
def scale(input_tensors, output_tensors, method="fro"):
    match method:
        case "absmax":
            input_tensors = np.float_(input_tensors)
            output_tensors = np.float_(output_tensors)

            # Normalize the dataset
            NValuesList = []

            for i, row in enumerate(input_tensors):
                NValuesList.append(max(np.abs(input_tensors[i,:])))
                input_tensors[i,:] = input_tensors[i,:]/NValuesList[i]
                output_tensors[i,:] = output_tensors[i,:]/NValuesList[i]
        
        case "fro":
            for i, row in enumerate(input_tensors):
                # Create a matrix from the data
                matrix = np.array([input_tensors[i,:]])

                # Calculate the Frobenius norm
                frobenius_norm = np.linalg.norm(matrix, 'fro')

                input_tensors[i,:] = input_tensors[i,:] / frobenius_norm
                output_tensors[i,:] = output_tensors[i,:] / frobenius_norm
            
    return input_tensors, output_tensors

# Normalize the tensors
input_tensors, output_tensors = scale(input_tensors, output_tensors, method="absmax")

# Combine the normalized tensors
normalized_data = np.concatenate((input_tensors, output_tensors), axis=1)

# Write the dataset to new .csv file in the same format as the original dataset
pd.DataFrame(normalized_data, columns=data.columns).to_csv("max_dOmegaRES100k.csv", index=False)
