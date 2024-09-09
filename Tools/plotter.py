import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df2 = pd.read_csv(r'deleteme\high_error_gradients.csv')
df = pd.read_csv(r"deleteme\dataset_training_unsampled.csv")

# Define the column names for the 9 elements of matrix A
matrix_a_columns = ['A11', 'A12', 'A13', 'A21', 'A22', 'A23', 'A31', 'A32', 'A33','ResVort']


# Initialize an empty list to store the matrices
matrix_list = []

# Initialize an empty list to store the vorticity values
vorticity_list = []

# Initialize an empty list to store the result values
rvalues_list = []

df = df[df['ResVort'] != 0]
df = df.head(50000)

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the 9 elements to form a 3x3 matrix
    matrix = np.array([
        [row['A11'], row['A12'], row['A13']],
        [row['A21'], row['A22'], row['A23']],
        [row['A31'], row['A32'], row['A33']]
    ])
    # Append the matrix to the list
    matrix_list.append(matrix)
    vorticity_list.append(row['ResVort'])

    # Spliting matrix A_ to symmetric (S) and antisymmetric (Ω) parts
    S = 0.5 * (matrix + matrix.T)
    Ω = 0.5 * (matrix - matrix.T)

    rvalues_list.append(np.linalg.norm(Ω, ord='fro')/np.linalg.norm(S, ord='fro'))


print(matrix_list[0])
print(vorticity_list[0])
print(rvalues_list[0])

min = min(rvalues_list)
print(min)


# Initialize additional lists for the df2 dataset
matrix_list2 = []
vorticity_list2 = []
rvalues_list2 = []

# Process the df2 dataset in the same way as the df dataset
for index, row in df2.iterrows():
    matrix = np.array([
        [row['A11'], row['A12'], row['A13']],
        [row['A21'], row['A22'], row['A23']],
        [row['A31'], row['A32'], row['A33']]
    ])
    matrix_list2.append(matrix)
    vorticity_list2.append(row['ResVort'])

    S = 0.5 * (matrix + matrix.T)
    Ω = 0.5 * (matrix - matrix.T)

    rvalues_list2.append(np.linalg.norm(Ω, ord='fro')/np.linalg.norm(S, ord='fro'))

# Plot the data for the df dataset in blue (this will be the default color)
plt.scatter(vorticity_list, rvalues_list, s=0.1)

# Plot the data for the df2 dataset in red
plt.scatter(vorticity_list2, rvalues_list2, s=1, color='red')

plt.xlabel('Vorticity')
plt.ylabel('R-values')
plt.title('Vorticity vs R-values')
plt.show()