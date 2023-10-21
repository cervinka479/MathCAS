import pyvista as pv
import pandas as pd
import numpy as np

# Load your dataset from a CSV file without headers
csv_file = "ANN_DataSet\.filtered_dataset_28K.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file, header=None)

# Extract X, Y, Z coordinates, and values using column indexes
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
z = df.iloc[:, 2].values
values = df.iloc[:, 3].values

# Create a structured grid from the coordinates
grid = pv.StructuredGrid()
grid.points = np.column_stack((x, y, z))

# Add the values to the point data
grid.point_arrays["Value"] = values

# Export the grid to a VTK file
vtk_filename = "your_output.vtk"  # Replace with your desired output file name
grid.save(vtk_filename)

print(f"VTK file '{vtk_filename}' has been created.")


'''import pyvista as pv

# Define the parameters of the cylinder grid
radius = 1.0
height = 5.0
resolution = 50

# Create a cylinder-shaped grid
cylinder = pv.Cylinder(radius=radius, height=height, resolution=resolution)

# Export the grid to a VTK file
output_file = "cylinder_grid.vtk"
cylinder.save(output_file)

print(f"{output_file} export completed")'''
