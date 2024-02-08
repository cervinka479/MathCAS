import scipy
from scipy.optimize import fmin
import numpy as np
import random

# Initial variables
A = np.array([[-2, 8, 3],
              [-9, -5, -5],
              [7, -1, 7]])

x0 = np.array([175, 45, 40])

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

def TripleDecomposition():

    # Creating transformation matrix Q based on result from fmin function
    Q = CreateQMatrix(result)
    
    # Transforming matrix A into matrix A_
    A_ = Q @ A @ Q.T
    print(A_)

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
    return values

# Acquiring coordinates of maximum BRF via fmin function
result = fmin(GetFValue, x0, maxiter=1000)
print(result[0:3])

# Acquiring value of maximum BRF
MaxBRF = -(GetFValue(result))
print(MaxBRF)

# Triple decomposition
print(TripleDecomposition())