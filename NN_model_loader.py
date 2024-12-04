import torch
import torch.nn as nn
import pandas as pd

def load_and_test_model(model_path, dataset_path, task='regression', num_samples=10):
    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Infer the architecture
    input_size = None
    hidden_layers = []
    output_size = None

    for key in state_dict.keys():
        if 'weight' in key:
            layer_shape = state_dict[key].shape
            if input_size is None:
                input_size = layer_shape[1]
            hidden_layers.append(layer_shape[0])
            output_size = layer_shape[0]

    # Remove the last hidden layer size as it is the output layer size
    hidden_layers = hidden_layers[:-1]

    print(f"Inferred architecture: Input size = {input_size}, Hidden layers = {hidden_layers}, Output size = {output_size}")

    # Define the model architecture
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, task='class'):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            
            # Create hidden layers
            for hidden_size in hidden_layers:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.ReLU())
                input_size = hidden_size
            
            # Create output layer
            self.layers.append(nn.Linear(input_size, output_size))
            if task == 'class':
                self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Load the model
    def load_model(model_path, input_size, hidden_layers, output_size, task='class'):
        model = NeuralNetwork(input_size, hidden_layers, output_size, task)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model

    # Load the model
    model = load_model(model_path, input_size, hidden_layers, output_size, task)
    print(model)

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Extract the first num_samples data points
    X_test = df.iloc[:num_samples, :9].values
    y_true = df.iloc[:num_samples, 11].values

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    # Print true results and predictions
    print("True results:", y_true)
    print("Predictions:", y_pred.flatten())

# Example usage
load_and_test_model(
    model_path='NN_training_log/trial2_best_model.pth',
    dataset_path='deleteme/dataset3D_50K_training_sampled.csv',
    task='regression',
    num_samples=10
)