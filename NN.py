import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nnArch(io=[9,1], hl=[12]):
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

def objective(trial):
    # Sample hyperparameters
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_units = [trial.suggest_int(f'n_units_l{i}', 4, 128) for i in range(hidden_layers)]
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    # Create the model
    model = nnArch(io=[9, 1], hl=hidden_units)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dummy data (replace with your actual data)
    X_train = torch.randn(100, 9)
    y_train = torch.randint(0, 2, (100, 1)).float()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Use more epochs for actual training
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Validation (use actual validation data)
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_train), y_train).item()
    
    return val_loss

# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')