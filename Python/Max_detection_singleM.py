from scipy.optimize import minimize
from scipy.optimize import fmin
import numpy as np
import random

# Initial variables
A = np.array([[random.randint(-100, 100), random.randint(-100, 100), random.randint(-100, 100)], 
              [random.randint(-100, 100), random.randint(-100, 100), random.randint(-100, 100)], 
              [random.randint(-100, 100), random.randint(-100, 100), random.randint(-100, 100)]])

print(A)

α = 0
β = 0
γ = 0

αMax = 180
αStep = 5

βMax = 180
βStep = 5

γMax = 90
γStep = 5

x0 = np.array([α, β, γ])

previousBRF = 0

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
    print(A_)
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
    return values

for i in range(int(γMax/γStep)):
    for j in range(int(βMax/βStep)):
        for k in range(int(αMax/αStep)):
            x0 = np.array([α, β, γ])
            result = minimize(GetFValue, x0, method='BFGS')
            optimal_params = result.x
            if previousBRF == 0:
                previousBRF = -(GetFValue(optimal_params))
                print(previousBRF, α, β, γ)
            else:
                MaxBRF = -(GetFValue(optimal_params))
                if np.abs(previousBRF-MaxBRF)>= 1e-3:
                    print(MaxBRF, α, β, γ)
                previousBRF = MaxBRF
            
            α = α+αStep
        α = 0
        β = β+βStep
    α = 0
    β = 0
    γ = γ+γStep

print("Completed")