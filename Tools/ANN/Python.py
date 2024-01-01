def generate():
    import random
    import numpy as np

    Ux = 0.15464893
    Uy = -0.50260904
    Vx = 0.85056914

    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

def scale(input_tensors, output_tensors, method="fro"):
    import numpy as np
    import pandas as pd

    match method:
        case "absmax":
            input_tensors = np.float_(input_tensors)
            output_tensors = np.float_(output_tensors)

            # Normalize the dataset
            NValuesList = []

            NValuesList.append(max(np.abs(input_tensors[:])))
            input_tensors[:] = input_tensors[:]/NValuesList
            output_tensors = output_tensors/NValuesList
        
        case "fro":
            # Create a matrix from the data
            matrix = np.array([input_tensors])

            # Calculate the Frobenius norm
            frobenius_norm = np.linalg.norm(matrix, 'fro')
            print(frobenius_norm)

            input_tensors = input_tensors / frobenius_norm
            output_tensors = output_tensors / frobenius_norm
            
    return input_tensors, output_tensors

print(generate())
#print(scale(generate()[0:3],generate()[3]))