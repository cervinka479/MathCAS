import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the CSV dataset
dataset_path = 'temp_preprocessed.csv'  # Replace with the actual path
data = pd.read_csv(dataset_path)

# Extract the input velocity-gradient tensors (matrix A) and output normalized tensors
input_tensors = data.iloc[:, 3:12].values
output_tensors = data.iloc[:, 12:13].values

# Make the input tensor deviatoric
traces = input_tensors[:,0:1] + input_tensors[:,4:5] + input_tensors[:,8:9]
input_tensors[:,0:1] -= 1./3. * traces
input_tensors[:,4:5] -= 1./3. * traces
input_tensors[:,8:9] -= 1./3. * traces

NValuesList = []


# Normalize the dataset

for i, row in enumerate(input_tensors):
    NValuesList.append(max(np.abs(input_tensors[i,:])))
    input_tensors[i,:] = input_tensors[i,:]/NValuesList[i]
    output_tensors[i,:] = output_tensors[i,:]/NValuesList[i]

NValsFile = pd.DataFrame(NValuesList, columns=['Values'])
NValsFile.to_csv('NValsFileFull.csv', index=False)

'''input_scaler = MinMaxScaler()
normalized_input_tensors = input_scaler.fit_transform(input_tensors)

output_scaler = MinMaxScaler()
normalized_output_tensors = output_scaler.fit_transform(output_tensors)'''

#print(normalized_output_tensors)

#print(output_scaler.inverse_transform(normalized_output_tensors))


# Split the data into training, validation, and test sets
from sklearn.model_selection import train_test_split

train_input, test_input, train_output, test_output = train_test_split(
    input_tensors, output_tensors, test_size=0.2, random_state=42
)

train_input, val_input, train_output, val_output = train_test_split(
    train_input, train_output, test_size=0.1, random_state=42
)

'''print(output_scaler.inverse_transform(test_output[:10]))'''

# Now train_input, val_input, test_input contain the normalized input tensors,
# and train_output, val_output, test_output contain the normalized output tensors.

# You can use these datasets to train your neural network.

# Inverse normalize the dataset

'''print(test_output)'''

'''print(normalized_input_tensors[3200,:])
print(normalized_output_tensors[3200,:])'''