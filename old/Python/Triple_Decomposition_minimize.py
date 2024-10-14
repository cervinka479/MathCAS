from scipy.optimize import fmin
from scipy.optimize import minimize
import numpy as np
import pandas as pd

resVortList = []

# Load the CSV dataset
dataset_path = 'temp_preprocessed.csv'  # Replace with the actual path
data = pd.read_csv(dataset_path)

print(enumerate(data))

for i in range(1369256):
    print(i)
    # Initial variables
    A = np.array([[data.iloc[i,3], data.iloc[i,6], data.iloc[i,9]],
                [data.iloc[i,4], data.iloc[i,7], data.iloc[i,10]],
                [data.iloc[i,5], data.iloc[i,8], data.iloc[i,11]]])

    x0 = np.array([11, 11, 11])

    # Function
    def CreateQMatrix(params):
        
        α, β, γ = params
        
        # Convert DEG to RAD
        α = np.deg2rad(α)
        β = np.deg2rad(β)
        γ = np.deg2rad(γ)

        # Creating transformation matrix Q
        Q = np.array([[np.cos(α)*np.cos(β)*np.cos(γ)-np.sin(α)*np.sin(γ), np.sin(α)*np.cos(β)*np.cos(γ)+np.cos(α)*np.sin(γ), -np.sin(β)*np.cos(γ)],
                    [-np.cos(α)*np.cos(β)*np.sin(γ)-np.sin(α)*np.cos(γ), -np.sin(α)*np.cos(β)*np.sin(γ)+np.cos(α)*np.cos(γ), np.sin(β)*np.sin(γ)],
                    [np.cos(α)*np.sin(β), np.sin(α)*np.sin(β), np.cos(β)]])
        
        return Q

    def GetFValue(params):

        Q = CreateQMatrix(params)

        # Transforming matrix A into matrix A_
        A_ = Q @ A @ Q.T

        # Spliting matrix A_ to symmetric (S) and antisymmetric (Ω) parts
        S = 0.5 * (A_ + A_.T)
        Ω = 0.5 * (A_ - A_.T)

        # Acquiring BRF
        f = -(np.abs(S[0, 1] * Ω[0, 1]) + np.abs(S[0, 2] * Ω[0, 2]) + np.abs(S[1, 2] * Ω[1, 2]))
        return f

    def TripleDecomposition(params):

        # Creating transformation matrix Q based on result from fmin function
        Q = CreateQMatrix(params)
        
        # Transforming matrix A into matrix A_
        A_ = Q @ A @ Q.T
        '''print(A_)'''
        #print(Q.T)

        # Splitting into ResidualTensor and SheerTensor
        ResidualTensor = np.array([[A_[0, 0], np.sign(A_[0, 1])*min(np.abs(A_[0, 1]),np.abs(A_[1, 0])), np.sign(A_[0, 2])*min(np.abs(A_[0, 2]),np.abs(A_[2, 0]))],
                                [np.sign(A_[1, 0])*min(np.abs(A_[0, 1]),np.abs(A_[1, 0])), A_[1, 1], np.sign(A_[1, 2])*min(np.abs(A_[1, 2]),np.abs(A_[2, 1]))],
                                [np.sign(A_[2, 0])*min(np.abs(A_[0, 2]),np.abs(A_[2, 0])), np.sign(A_[2, 1])*min(np.abs(A_[1, 2]),np.abs(A_[2, 1])), A_[2, 2]]])

        ShearTensor = A_ - ResidualTensor

        # Spliting ResidualTensor to symmetric (S) and antisymmetric (Ω) parts
        S = 0.5 * (ResidualTensor + ResidualTensor.T)
        Ω = 0.5 * (ResidualTensor - ResidualTensor.T)

        # Decomposition of individual elements
        values = [np.linalg.norm(S, ord='fro'), np.linalg.norm(Ω, ord='fro'), np.linalg.norm(ShearTensor, ord='fro')]
        return values[1]

    # Acquiring coordinates of maximum BRF via fmin function
    result = minimize(GetFValue, x0, method='BFGS')
    optimal_params = result.x  # The optimized parameters (α, β, γ) are now stored in result.x

    # Acquiring value of maximum BRF
    MaxBRF = -result.fun  # result.fun contains the value of the objective function at the minimum
    '''print("Optimal Parameters (α, β, γ):", optimal_params)
    print("Maximum BRF:", MaxBRF)'''

    resVortList.append(TripleDecomposition(optimal_params))
    resVort = pd.DataFrame(resVortList, columns=['Values'])
    resVort.to_csv('resVort.csv', index=False)

print("completed")