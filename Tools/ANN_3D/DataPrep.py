# Writes an output file #
def generateDataset(filename, num_items, nonzero=False):
    import csv

    zero_vorticity_count = 0

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A11","A12","A13","A21","A22","A23","A31","A32","A33","vorticity"])
        for _ in range(num_items):
            row = [*generate(nonzero)]
            writer.writerow([*row])
            percentage = (_ / num_items) * 100
            print(f"{percentage:.2f}% ({_}/{num_items})\n", end="")

            # Check if vorticity is zero
            if row[-1] == 0:
                zero_vorticity_count += 1

    print("Dataset generated: "+filename)

    # Print the ratio of zero and non-zero vorticity values
    print(f"zero values percentage: {zero_vorticity_count*100/num_items:.2f}")

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

def extract(path, i=[1,2], o=[3,3], limit=0, deviatoric=False):
    import pandas as pd

    # Load the CSV dataset
    data = pd.read_csv(path)

    if limit == 0:
        # Extract the input velocity-gradient tensors (matrix A) and output normalized tensors
        input_tensors = data.iloc[:, i[0]-1:i[1]].values # [:, 3:12]
        output_tensors = data.iloc[:, o[0]-1:o[1]].values# [:, 12:13]
    else:
        # Extract the input velocity-gradient tensors (matrix A) and output normalized tensors
        input_tensors = data.iloc[:limit, i[0]-1:i[1]].values # [:, 3:12]
        output_tensors = data.iloc[:limit, o[0]-1:o[1]].values# [:, 12:13]


    if deviatoric==True:
        # Make the input tensor deviatoric
        traces = input_tensors[:,0:1] + input_tensors[:,4:5] + input_tensors[:,8:9]
        input_tensors[:,0:1] -= 1./3. * traces
        input_tensors[:,4:5] -= 1./3. * traces
        input_tensors[:,8:9] -= 1./3. * traces

    return input_tensors,output_tensors

def split(input_tensors, output_tensors):
    from sklearn.model_selection import train_test_split

    train_input, val_input, train_output, val_output = train_test_split(
        input_tensors, output_tensors, test_size=0.1, random_state=42
    )

    return train_input, train_output, val_input, val_output

def scale(input_tensors, output_tensors, method="fro", class_labels=False):
    import numpy as np
    import pandas as pd

    match method:
        case "absmax":
            input_tensors = np.float_(input_tensors)
            output_tensors = np.float_(output_tensors)

            # Normalize the dataset
            NValuesList = []

            for i, row in enumerate(input_tensors):
                NValuesList.append(max(np.abs(input_tensors[i,:])))
                input_tensors[i,:] = input_tensors[i,:]/NValuesList[i]
                if class_labels==False:
                    output_tensors[i,:] = output_tensors[i,:]/NValuesList[i]
        
        case "fro":
            for i, row in enumerate(input_tensors):
                # Create a matrix from the data
                matrix = np.array([input_tensors[i,:]])

                # Calculate the Frobenius norm
                frobenius_norm = np.linalg.norm(matrix, 'fro')

                input_tensors[i,:] = input_tensors[i,:] / frobenius_norm
                if class_labels==False:
                    output_tensors[i,:] = output_tensors[i,:] / frobenius_norm
            
    return input_tensors, output_tensors

def inverseScale(input_tensors, predictions, method="fro"):
    import numpy as np
    import pandas as pd

    match method:
        case "absmax":
            input_tensors = np.float_(input_tensors)
            predictions = np.float_(predictions)

            NValuesList = []

            for i, row in enumerate(input_tensors):
                NValuesList.append(max(np.abs(input_tensors[i,:])))
                predictions[i,:] = predictions[i,:]*NValuesList[i]
        
        case "fro":
            
            for i, row in enumerate(input_tensors):
                # Create a matrix from the data
                matrix = np.array([input_tensors[i,:]])

                # Calculate the Frobenius norm
                frobenius_norm = np.linalg.norm(matrix, 'fro')

                predictions[i,:] = predictions[i,:] * frobenius_norm

    return predictions

def generate(nonzero2D=False):
    import random
    import numpy as np

    if nonzero2D==True:
        ωRES = 0
        while ωRES == 0:
            Ux = random.random()*2-1
            Uy = random.random()*2-1
            Vx = random.random()*2-1

            s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
            ω = (Vx-Uy)/2

            if np.abs(s) <= np.abs(ω):
                ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
            else:
                ωRES = 0
    else:
        A11 = random.random()*2-1
        A12 = random.random()*2-1
        A13 = random.random()*2-1
        A21 = random.random()*2-1
        A22 = random.random()*2-1
        A23 = random.random()*2-1
        A31 = random.random()*2-1
        A32 = random.random()*2-1
        A33 = -A11-A22

        import BruteforceMethod as bf

        A = np.array([[A11, A12, A13],
                    [A21, A22, A23],
                    [A31, A32, A33]])
        
        vorticity = bf.getVorticity(A, Step=5)


    return A11, A12, A13, A21, A22, A23, A31, A32, A33, vorticity

def toBinary(filename):
    import pandas as pd
    
    # Load the dataset
    df = pd.read_csv(filename)

    # Replace all non-zero values in the 'vorticity' column with 1
    df.loc[df['ResVort'] != 0, 'ResVort'] = 1

    # Save the modified dataset to a new CSV file
    df.to_csv('bin-'+filename, index=False)

#generateDataset("dataset3D.csv", 100)

#toBinary("dataset3D10k.csv")

'''
print(extract(path="dataset3D10k.csv",i=[1,9],o=[10,10],limit=10))
absmaxScaledData = scale(extract(path="dataset3D10k.csv",i=[1,9],o=[10,10],limit=10)[0],extract(path="dataset3D10k.csv",i=[1,9],o=[10,10],limit=10)[1], method="absmax")
print(absmaxScaledData)
print(inverseScale(extract(path="dataset3D10k.csv",i=[1,9],o=[10,10],limit=10)[0],absmaxScaledData[1], method="absmax"))
'''