import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import PrepData

features = PrepData.features
labels = PrepData.labels

test_features = PrepData.test_features

print(test_features)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(9, 16)
        self.output_layer = nn.Linear(16, 4)  # 3 for coordinates, 1 for F value

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden_layer(x))
        output = self.output_layer(x)
        return output

# Create an instance of the neural network
model = NeuralNetwork()
print(model)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can experiment with different optimizers and learning rates

# Define a function for training the model
def train_model(model, features, labels, epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Train the model
train_model(model, features, labels)

def predict(model, features):
    with torch.no_grad():
        predictions = model(features)
    return predictions.numpy()  # Convert predictions to NumPy array

# Assuming 'test_features' is a NumPy array containing 3x3 matrices for testing.
test_features = torch.tensor(test_features, dtype=torch.float32)
predictions = predict(model, test_features)
print(predictions)
