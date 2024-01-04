def generate(distribution='cube'):
    import random
    import numpy as np

    if distribution == 'cube':
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
    
    elif distribution == 'sphere':
        # Generate two random numbers
        theta = 2 * np.pi * random.random()  # Uniform from [0, 2π)
        phi = np.arccos(2 * random.random() - 1)  # Uniform from [0, π)

        # Convert spherical coordinates to Cartesian coordinates
        Ux = np.sin(phi) * np.cos(theta)
        Uy = np.sin(phi) * np.sin(theta)
        Vx = np.cos(phi)

    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

def generateDataset(filename, num_items, distribution='cube'):
    import csv

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ux","Uy","Vx","omegaRES"])
        for _ in range(num_items):
            writer.writerow([*generate(distribution)])

    print("Dataset generated: "+filename)

generateDataset("sphere_dOmegaRES100k.csv", 100000, distribution='sphere')