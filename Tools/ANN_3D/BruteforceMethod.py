import numpy as np
import pyvista as pv
import csv
import random


# Input variables
A = np.array([[-0.22, 0.8, 0.22],
              [-0.82, -0.42, -0.22],
              [0.82, -0.22, 0.64]])

def bruteforceBRF(matrix, Step=5, Output=False, Info=False):

    α = 0
    β = 0
    γ = 0

    αMax = 180
    αStep = Step

    βMax = 180
    βStep = Step

    γMax = 90
    γStep = Step

    data = []

    def GetFValue(α, β, γ):
        
        # Convert DEG to RAD
        α = np.deg2rad(α)
        β = np.deg2rad(β)
        γ = np.deg2rad(γ)

        # Creating matrix Q
        Q = np.array([[np.cos(α)*np.cos(β)*np.cos(γ)-np.sin(α)*np.sin(γ), np.sin(α)*np.cos(β)*np.cos(γ)+np.cos(α)*np.sin(γ), -np.sin(β)*np.cos(γ)],
                    [-np.cos(α)*np.cos(β)*np.sin(γ)-np.sin(α)*np.cos(γ), -np.sin(α)*np.cos(β)*np.sin(γ)+np.cos(α)*np.cos(γ), np.sin(β)*np.sin(γ)],
                    [np.cos(α)*np.sin(β), np.sin(α)*np.sin(β), np.cos(β)]])

        # Creating matrix A_
        A_ = Q @ A @ Q.T

        # Spliting matrix A_ to symmetric (S) and antisymmetric (Ω) parts
        S = 0.5 * (A_ + A_.T)
        Ω = 0.5 * (A_ - A_.T)

        # Acquiring f value
        f = np.abs(S[0, 1] * Ω[0, 1]) + np.abs(S[0, 2] * Ω[0, 2]) + np.abs(S[1, 2] * Ω[1, 2])
        return f

    maxValues = [[0, 0, 0, 0]]

    for i in range(int(γMax/γStep)):
        for j in range(int(βMax/βStep)):
            for k in range(int(αMax/αStep)):
                f_value = GetFValue(α, β, γ)
                data.append([α, β, γ, f_value])
                if f_value > maxValues[0][3]:
                    maxValues = []
                    maxValues.append([α, β, γ, f_value])
                elif f_value == maxValues[0][3]:
                    maxValues.append([α, β, γ, f_value])
                
                α = α+αStep
            α = 0
            β = β+βStep
        α = 0
        β = 0
        γ = γ+γStep

    data_arr = np.array(data)

    # Extract coordinates and F values
    coordinates = data_arr[:, :3]
    f_values = data_arr[:, 3]

    grid = pv.StructuredGrid()
    grid.points = coordinates
    grid.point_data["FValues"] = f_values

    grid.dimensions = (int(αMax/αStep), int(βMax/βStep), int(γMax/γStep))
    grid.spacing = (αStep, βStep, γStep)
    grid.origin = (0, 0, 0)

    if Output == True:
        # Export the grid to VTK file
        output_file = "output.vtk"
        grid.save(output_file)

        print("output.vtk export completed")

    if Info == True:
        print(np.array(maxValues))
    
    return maxValues[0][0:3]

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

def TripleDecomposition(result):

    # Creating transformation matrix Q based on result from fmin function
    Q = CreateQMatrix(result)
    
    # Transforming matrix A into matrix A_
    A_ = Q @ A @ Q.T

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

print(TripleDecomposition(bruteforceBRF(A, Step=5)))