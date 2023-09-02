import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import DataPrep.normalizeDataset as Data
import torchvision.models as models

# Load your trained model (replace 'trained_model_resVort.pth' with your model's path)
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('trained_model_resVort.pth'))
model.eval()

# Script the model
scripted_model = torch.jit.script(model)

# Save the scripted model to a file
scripted_model.save("scripted_model.pt")


