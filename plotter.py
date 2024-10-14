import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
'''df2 = pd.read_csv(r'deleteme\high_error_gradients.csv')'''
df = pd.read_csv(r"deleteme\dataset3D_50K_training_sampled.csv")

# Define the column names for the 9 elements of matrix A
matrix_a_columns = ['A11', 'A12', 'A13', 'A21', 'A22', 'A23', 'A31', 'A32', 'A33','ResVort']


# Initialize an empty list to store the matrices
matrix_list = []

# Initialize an empty list to store the vorticity values
vorticity_list = []

# Initialize an empty list to store the result values
rvalues_list = []

# Initialize an empty list to store the delta values
deltavalues_list = []

# Initialize an empty list to store the lambda2 values
lambda2values_list = []

vortex = 0
negative = 0
positive = 0

#df = df[df['ResVort'] != 0]
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

    rvalues_list.append((np.linalg.norm(Ω, ord='fro')/np.linalg.norm(S, ord='fro'))-1)

    # Calculate the delta value
    Q = (matrix[0][0]*matrix[1][1]) + (matrix[1][1]*matrix[2][2]) + (matrix[0][0]*matrix[2][2]) - (matrix[0][1]*matrix[1][0]) - (matrix[1][2]*matrix[2][1]) - (matrix[0][2]*matrix[2][0])
    R = -(matrix[0][2]*matrix[1][1]*matrix[2][0]) + (matrix[0][1]*matrix[1][2]*matrix[2][0]) + (matrix[0][2]*matrix[1][0]*matrix[2][1]) - (matrix[0][0]*matrix[1][2]*matrix[2][1]) - (matrix[0][1]*matrix[1][0]*matrix[2][2]) + (matrix[0][0]*matrix[1][1]*matrix[2][2])

    deltavalue = (Q/3)**3 + (R/2)**2
    deltavalues_list.append(deltavalue)

    if row['ResVort'] != 0:
        vortex += 1
        if deltavalue < 0:
            negative += 1
        else:
            positive += 1

    # Calculate the lambda2 criterion
    JHtensor = S@S + Ω@Ω
    eigenvalues = np.linalg.eigvals(JHtensor)
    lambda2 = np.sort(eigenvalues)[1]
    lambda2values_list.append(-lambda2)
    if index == 0:
        print(matrix)
        print(eigenvalues)
        print(lambda2)
        print("- - - - - - - - - -")

    counter = 0
    if lambda2 < 0 and deltavalue < 0:
        counter =+1

print(matrix_list[0])
print(vorticity_list[0])
print(rvalues_list[0])
print(deltavalues_list[0])
print(lambda2values_list[0])

print(counter)

'''min = min(rvalues_list)
print(min)'''

'''print(vortex)
print(negative)
print(positive)'''

'''# Initialize additional lists for the df2 dataset
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

    rvalues_list2.append(np.linalg.norm(Ω, ord='fro')/np.linalg.norm(S, ord='fro'))'''

'''# Normalize the values to range between 0 and 1
norm_deltavalues = [float(i) / max(deltavalues_list) for i in deltavalues_list]
norm_deltavalues = [(i + 1) / 2 for i in norm_deltavalues]'''

# Add zero axes
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.scatter(deltavalues_list, lambda2values_list, s=1)

plt.xlabel('delta')
plt.ylabel('-lambda2')
plt.show()