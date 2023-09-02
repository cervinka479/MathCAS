import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy
import DataPrep.normalizeDataset as Data
import matplotlib.pyplot as plt


def HiddenLayer1(hidden_size = 64, learning_rate = 0.001, num_epochs = 200):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    # Create the neural network instance
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Print the model architecture
    print(model)



    # Define loss function (mean squared error)
    criterion = nn.MSELoss()

    # Define optimizer (e.g., Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.FloatTensor(Data.train_input)
    train_output_tensor = torch.FloatTensor(Data.train_output)
    val_input_tensor = torch.FloatTensor(Data.val_input)
    val_output_tensor = torch.FloatTensor(Data.val_output)

    # Training loop
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_input_tensor)
        
        # Compute loss
        loss = criterion(outputs, train_output_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())


    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_input_tensor)
        val_loss = criterion(val_predictions, val_output_tensor)
        
    print(f'Validation Loss: {val_loss.item():.4f}')

    # Convert predictions and true values to numpy arrays for evaluation
    val_predictions_np = val_predictions.numpy()
    val_output_np = val_output_tensor.numpy()

    return loss_values

def HiddenLayer2(hidden_size1 = 64, hidden_size2 = 32, learning_rate = 0.001, num_epochs = 200):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x


    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    # Create the neural network instance
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

    # Print the model architecture
    print(model)



    # Define loss function (mean squared error)
    criterion = nn.MSELoss()

    # Define optimizer (e.g., Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.FloatTensor(Data.train_input)
    train_output_tensor = torch.FloatTensor(Data.train_output)
    val_input_tensor = torch.FloatTensor(Data.val_input)
    val_output_tensor = torch.FloatTensor(Data.val_output)

    # Training loop
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_input_tensor)
        
        # Compute loss
        loss = criterion(outputs, train_output_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())


    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_input_tensor)
        val_loss = criterion(val_predictions, val_output_tensor)
        
    print(f'Validation Loss: {val_loss.item():.4f}')

    # Convert predictions and true values to numpy arrays for evaluation
    val_predictions_np = val_predictions.numpy()
    val_output_np = val_output_tensor.numpy()

    return loss_values

def HiddenLayer3(hidden_size1 = 64, hidden_size2 = 32, hidden_size3 = 32, learning_rate = 0.001, num_epochs = 200):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size3, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
            return x


    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    # Create the neural network instance
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

    # Print the model architecture
    print(model)



    # Define loss function (mean squared error)
    criterion = nn.MSELoss()

    # Define optimizer (e.g., Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.FloatTensor(Data.train_input)
    train_output_tensor = torch.FloatTensor(Data.train_output)
    val_input_tensor = torch.FloatTensor(Data.val_input)
    val_output_tensor = torch.FloatTensor(Data.val_output)

    # Training loop
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_input_tensor)
        
        # Compute loss
        loss = criterion(outputs, train_output_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())


    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_input_tensor)
        val_loss = criterion(val_predictions, val_output_tensor)
        
    print(f'Validation Loss: {val_loss.item():.4f}')

    # Convert predictions and true values to numpy arrays for evaluation
    val_predictions_np = val_predictions.numpy()
    val_output_np = val_output_tensor.numpy()

    model_name = 'trained_model3.pth'
    torch.save(model.state_dict(), model_name)

    return loss_values

def HiddenLayer5(hidden_size1 = 64, hidden_size2 = 32, hidden_size3 = 32, hidden_size4 = 32, hidden_size5 = 32, learning_rate = 0.001, num_epochs = 200):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size3, hidden_size4)
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(hidden_size4, hidden_size5)
            self.relu5 = nn.ReLU()
            self.fc6 = nn.Linear(hidden_size5, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
            x = self.relu4(x)
            x = self.fc5(x)
            x = self.relu5(x)
            x = self.fc6(x)
            return x


    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    # Create the neural network instance
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size)

    # Print the model architecture
    print(model)



    # Define loss function (mean squared error)
    criterion = nn.MSELoss()

    # Define optimizer (e.g., Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.FloatTensor(Data.train_input)
    train_output_tensor = torch.FloatTensor(Data.train_output)
    val_input_tensor = torch.FloatTensor(Data.val_input)
    val_output_tensor = torch.FloatTensor(Data.val_output)

    # Training loop
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_input_tensor)
        
        # Compute loss
        loss = criterion(outputs, train_output_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())


    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_input_tensor)
        val_loss = criterion(val_predictions, val_output_tensor)
        
    print(f'Validation Loss: {val_loss.item():.4f}')

    # Convert predictions and true values to numpy arrays for evaluation
    val_predictions_np = val_predictions.numpy()
    val_output_np = val_output_tensor.numpy()

    model_name = 'trained_model5.pth'
    torch.save(model.state_dict(), model_name)

    return loss_values

def UseTrainedModel3(hidden_size1 = 1000, hidden_size2 = 1000, hidden_size3 = 1000, hidden_size4 = 1000, hidden_size5 = 1000):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size3, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
            return x


    # Define input, hidden, and output sizes
    input_size = 9  # Number of input features (velocity-gradient tensor)
    output_size = 1  # Number of output features (normalized S, Ω, shear tensor)

    model_name = 'trained_model3.pth'
    loaded_model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode

    # Prepare the test input data
    test_input_tensor = torch.FloatTensor(Data.normalized_input_tensors)

    # Pass the test input data through the model
    with torch.no_grad():
        predictions_normalized = loaded_model(test_input_tensor)
        
    # Convert the predictions to a numpy array
    predictions_normalized = predictions_normalized.numpy()

    # Inverse normalize the predictions to obtain actual output values
    predictions_actual = Data.output_scaler.inverse_transform(predictions_normalized)

    # Convert the predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions_actual, columns=['Predicted_Ω'])

    # Save the DataFrame to a .csv file
    predictions_csv_file = 'predicted_omega.csv'
    predictions_df.to_csv(predictions_csv_file, index=False)

    print(f'Predicted Ω values saved to {predictions_csv_file}')



    '''# Define the expected input shape (replace with your model's input shape)
    input_shape = (1, 9)  # Example shape: (batch_size, input_size)

    # Generate random example input data
    example_input = torch.randn(input_shape)

    # Convert the model to TorchScript format
    scripted_model = torch.jit.trace(loaded_model, example_input)

    # Save the TorchScript model to a file
    scripted_model_name = 'scripted_model.pt'  # Choose a file name for your scripted model
    scripted_model.save(scripted_model_name)'''



    # Print the first 10 predictions
    print(predictions_actual[:10])

    # Convert actual_outputs and predictions_actual to PyTorch tensors
    actual_outputs_tensor = torch.FloatTensor(Data.normalized_output_tensors)
    predictions_actual_tensor = torch.FloatTensor(predictions_actual)

    # Calculate the mean of actual_outputs
    mean_actual = torch.mean(actual_outputs_tensor)

    # Calculate the total sum of squares (TSS) and residual sum of squares (RSS)
    tss = torch.sum((actual_outputs_tensor - mean_actual)**2)
    rss = torch.sum((actual_outputs_tensor - predictions_actual_tensor)**2)

    # Calculate R-squared
    r2 = 1 - (rss / tss)

    print(f'R-squared: {r2:.4f}')

UseTrainedModel3(hidden_size1 = 1000, hidden_size2 = 1000, hidden_size3 = 1000, hidden_size4 = 1000, hidden_size5 = 1000)