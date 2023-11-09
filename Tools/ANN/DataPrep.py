# Writes an output file #
def removeEndCommas(path):
    import pandas as pd

    # Read the CSV file and preprocess to remove trailing commas
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = [line.rstrip(',\n') + '\n' for line in lines]

    # Write the preprocessed lines to a new temporary file
    with open("fixedData.csv", 'w') as temp_file:
        temp_file.writelines(lines)

    print("Fixed")

# Writes an output file #
def filter(path, compare=False):
    import pandas as pd

    # Read the CSV file
    original_df = pd.read_csv(path)

    # Add your own masks
    mask = (original_df['X'] >= 5.0) & (original_df['X'] <= 5.5)
    #mask = mask & (original_df['Y'] ** 2 + original_df['Z'] ** 2 <= 4)
    #mask = mask & ((original_df['A11']**2 + original_df['A21']**2 + original_df['A31']**2 + original_df['A12']**2 + original_df['A22']**2 + original_df['A32']**2 + original_df['A13']**2 + original_df['A23']**2 + original_df['A33']**2)>=8e-03)
    #mask = mask & (original_df['ResVort'] == 0)

    #print((original_df['A11']**2 + original_df['A21']**2 + original_df['A31']**2 + original_df['A12']**2 + original_df['A22']**2 + original_df['A32']**2 + original_df['A13']**2 + original_df['A23']**2 + original_df['A33']**2))

    filtered_data = original_df[mask]

    #(filtered_data['A11']**2 + filtered_data['A21']**2 + filtered_data['A31']**2 + filtered_data['A12']**2 + filtered_data['A22']**2 + filtered_data['A32']**2 + filtered_data['A13']**2 + filtered_data['A23']**2 + filtered_data['A33']**2).to_csv("filtered_dataset.csv", index=False, float_format='%.13E')

    # Export the filtered dataset to a new CSV file
    filtered_data.to_csv("filteredData.csv", index=False, float_format='%.13E')
    
    if compare==True:
        print(original_df)
        print("- - - - - - - - - -")
        print(filtered_data)
    else:
        print("Filtered")

def extract(path, i=[1,2], o=[3,3], deviatoric=False):
    import pandas as pd

    # Load the CSV dataset
    data = pd.read_csv(path)

    # Extract the input velocity-gradient tensors (matrix A) and output normalized tensors
    input_tensors = data.iloc[:, i[0]-1:i[1]].values # [:, 3:12]
    output_tensors = data.iloc[:, o[0]-1:o[1]].values# [:, 12:13]


    if deviatoric==True:
        # Make the input tensor deviatoric
        traces = input_tensors[:,0:1] + input_tensors[:,4:5] + input_tensors[:,8:9]
        input_tensors[:,0:1] -= 1./3. * traces
        input_tensors[:,4:5] -= 1./3. * traces
        input_tensors[:,8:9] -= 1./3. * traces

    return input_tensors,output_tensors

def split(input_tensors, output_tensors):
    from sklearn.model_selection import train_test_split

    train_input, test_input, train_output, test_output = train_test_split(
        input_tensors, output_tensors, test_size=0.2, random_state=42
    )

    train_input, val_input, train_output, val_output = train_test_split(
        train_input, train_output, test_size=0.1, random_state=42
    )

    return train_input, train_output, val_input, val_output, test_input, test_output

def normalize(input_tensors, output_tensors):
    import numpy as np
    import pandas as pd
    
    # Normalize the dataset
    NValuesList = []

    for i, row in enumerate(input_tensors):
        NValuesList.append(max(np.abs(input_tensors[i,:])))
        input_tensors[i,:] = input_tensors[i,:]/NValuesList[i]
        output_tensors[i,:] = output_tensors[i,:]/NValuesList[i]

    return input_tensors, output_tensors

def inverseNormalize(input_tensors, predictions):
    import numpy as np
    import pandas as pd

    NValuesList = []

    for i, row in enumerate(input_tensors):
        NValuesList.append(max(np.abs(input_tensors[i,:])))
        predictions[i,:] = predictions[i,:]*NValuesList[i]

    return predictions

#
print(split(*extract("test.csv",i=[1,2],o=[3,3])))