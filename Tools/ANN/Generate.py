def nnArchClass(io=[9,1], hl=[12]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class NeuralNetwork(nn.Module):
        def __init__(self, hl):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            input_size = io[0]
            
            # Create hidden layers
            for hidden_size in hl:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                input_size = hidden_size
            
            # Create output layer
            self.layers.append(nn.Linear(input_size, io[1]))
            self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create the neural network instance
    model = NeuralNetwork(hl)
    return model

def generate(shape='random',classifier="path",arch=nnArchClass()):
    import random
    import numpy as np
    import torch

    def cube():
        # Choose a random face of the cube (6 faces)
        face = random.choice(['x', 'y', 'z', '-x', '-y', '-z'])

        # Generate two random numbers for the chosen face
        U = random.random()*2-1
        V = random.random()*2-1

        # Depending on the chosen face, assign the random numbers to Ux, Uy, Vx
        if face == 'x':
            Ux, Uy, Vx = 1, U, V
        elif face == 'y':
            Ux, Uy, Vx = U, 1, V
        elif face == 'z':
            Ux, Uy, Vx = U, V, 1
        elif face == '-x':
            Ux, Uy, Vx = -1, U, V
        elif face == '-y':
            Ux, Uy, Vx = U, -1, V
        elif face == '-z':
            Ux, Uy, Vx = U, V, -1
        
        return Ux,Uy,Vx

    def sphere():
        # Generate two random numbers
        theta = 2 * np.pi * random.random()  # Uniform from [0, 2π)
        phi = np.arccos(2 * random.random() - 1)  # Uniform from [0, π)

        # Convert spherical coordinates to Cartesian coordinates
        Ux = np.sin(phi) * np.cos(theta)
        Uy = np.sin(phi) * np.sin(theta)
        Vx = np.cos(phi)

        return Ux,Uy,Vx

    def rand():
        Ux = random.random()*2-1
        Uy = random.random()*2-1
        Vx = random.random()*2-1

        return Ux,Uy,Vx

    if shape == 'cube':
        Ux, Uy, Vx = cube()
    
    elif shape == 'sphere':
        Ux, Uy, Vx = sphere()

    elif shape == 'random':
        Ux, Uy, Vx = rand()

    if classifier != "path":
        # Load the saved model
        arch.load_state_dict(torch.load(classifier))  # Load the saved parameters
        arch.eval()  # Set the model to evaluation mode

        features = [Ux, Uy, Vx]
        features = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            output = arch(features)
            print(output)
            prediction = torch.where(output > 0.5, 1.0, 0.0)
    
    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

def generateDataset(filename, num_items, shape='random',classifier="path",arch=nnArchClass()):
    import csv

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ux","Uy","Vx","omegaRES"])
        for _ in range(num_items):
            writer.writerow([*generate(shape,classifier)])

    print("Dataset generated: "+filename)

def nnGenerate(classifier="path",arch=nnArchClass()):
    import random
    import numpy as np
    import torch

    # Load the saved model
    arch.load_state_dict(torch.load(classifier))  # Load the saved parameters
    arch.eval()  # Set the model to evaluation mode

    nonzero = False
    while nonzero == False:
        Ux = random.random()*2-1
        Uy = random.random()*2-1
        Vx = random.random()*2-1
        
        features = [Ux, Uy, Vx]
        features = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            output = arch(features)
            #print(output)
            nonzero = torch.where(output > 0.5, True, False)

    # calculate ωRES
    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

def nnGenerateDataset(filename, num_items, classifier="path",arch=nnArchClass()):
    import csv

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ux","Uy","Vx","omegaRES"])
        for _ in range(num_items):
            writer.writerow([*nnGenerate(classifier,arch)])

    print("Dataset generated: "+filename)

#generateDataset("test1k.csv", 1000, shape='random')

nnGenerateDataset("3test8k.csv", 8000, classifier="classTest3_VL{1.462e-02}.pth", arch=nnArchClass(io=[3,1], hl=[32,16]))