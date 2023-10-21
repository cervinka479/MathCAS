import pyvista as pv
import pandas as pd
import numpy as np

αMax = 180
αStep = 2

βMax = 180
βStep = 2

γMax = 90
γStep = 2

# Load your dataset from a CSV file without headers
csv_file = "ANN_DataSet\.filtered_dataset_28K.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file, header=None)

# Extract coordinates and F values
coordinates = df.iloc[:, 0:3].values
values = df.iloc[:, 13].values

grid = pv.StructuredGrid()
grid.points = coordinates
grid.point_data["Values"] = values

grid.dimensions = (int(αMax/αStep), int(βMax/βStep), int(γMax/γStep))
grid.spacing = (αStep, βStep, γStep)
grid.origin = (0, 0, 0)

# Export the grid to VTK file
output_file = "output_cylinder.vtk"
grid.save(output_file)

print("output.vtk export completed")

