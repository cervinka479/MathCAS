import DataPrep
import torch
import time

# Check if CUDA is available
print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nnArch(io=[9,1], hl=[12]):
    import torch.nn as nn
    
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

def nnTrain(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="bce",batch_size=32, epochs=50, save="test", visualize=True, cli=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import copy

    # Unpack your dataset
    train_input, train_output, val_input, val_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32).to(device), torch.tensor(train_output, dtype=torch.float32).to(device))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32).to(device), torch.tensor(val_output, dtype=torch.float32).to(device))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size)
    val_loader = DataLoader(val_data, batch_size)

    print(model)
    model.to(device)

    match optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), learningRate)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), learningRate)

    match criterion:
        case "mse":
            criterion=torch.nn.MSELoss()
        case "bce":
            criterion=torch.nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accuracy = []
    models = []
    minimal_val_loss = "x"
    saveNum = 1
    total_time = 0

    # Training loop
    while True:
        for epoch in range(epochs):
            start_time = time.time()  # Start time of the epoch
            train_loss = 0
            val_loss = 0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation at the end of the epoch
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                for inputs, targets in val_loader:

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Calculate accuracy
                    predicted = torch.round(outputs.data)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                val_loss /= len(val_loader)  # Average validation loss
                val_acc = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracy.append(val_acc)         
            
            end_time = time.time()  # End time of the epoch
            elapsed_time = end_time - start_time  # Calculate elapsed time
            total_time += elapsed_time  # Add elapsed time to total time

            avg_time_per_epoch = total_time / (epoch + 1)  # Calculate average time per epoch
            remaining_epochs = epochs - epoch - 1  # Calculate remaining epochs
            remaining_time = avg_time_per_epoch * remaining_epochs  # Estimate remaining time

            remaining_time_formatted = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}, Estimated Remaining Time: {remaining_time_formatted}')
            
            model.train()  # Set the model back to training mode
            
            models.append(copy.deepcopy(model.state_dict()))
            
            # Save trained model
            if minimal_val_loss == "x":
                minimal_val_loss = val_loss
            elif val_loss < minimal_val_loss:
                bestModel = copy.deepcopy(model.state_dict())
                model.eval()
                example_input = torch.randn(1, 9).to(device)  # Replace with an example input to the model
                bestModelTorchscript = torch.jit.trace(model, example_input)
                minimal_val_loss = val_loss
        
        print("minimal validation loss: "+str(minimal_val_loss))
        
        # Visualize the training process
        if visualize == True:
            nnVisualize(train_losses, val_losses, val_accuracy)
        
        # Interactive Training CLI
        if cli == True:
            while True:
                try:
                    answer = input("\nTraining CLI:\nexit - to break the training loop\nshow - to show the visualization\nnext () - (number) to train further for number of epochs\nsave () - (number) to save the model from that epoch, (min) to save the model with the lowest validation loss\n\n")
                    if answer[0:4] == "exit":
                        cli = False
                        break
                    elif answer[0:4] == "show":
                        nnVisualize(train_losses, val_losses, val_accuracy)
                    elif answer[0:4] == "next":
                        epochs = int(answer[5:])
                        break
                    elif answer[0:4] == "save":
                        if answer[5:] == "min":
                            savePath = save+str(saveNum)+"_VL{"+str("{:.3e}".format(minimal_val_loss))+"}.pth"
                            torch.save(bestModel, savePath)
                        else:
                            savePath = save+str(saveNum)+"_VL{"+str("{:.3e}".format(val_losses[int(answer[5:])]))+"}.pth"
                            torch.save(models[int(answer[5:])-1], savePath)
                        saveNum = saveNum + 1
                        print("saved model: "+savePath)
                except Exception:
                    print("E: Input is in wrong format")
        
        if cli == False:
            savePath = save+str(saveNum)+"_VL{"+str("{:.3e}".format(minimal_val_loss))+"}.pth"
            torch.save(bestModel, savePath)
            # Save the TorchScript model
            torch.jit.save(bestModelTorchscript, f"{save}_scripted.pt")
            break

    #return train_losses, val_losses
    return minimal_val_loss

def nnPredict(loadModel, testDataset, model=nnArch(), output=False):
    import torch
    import pandas as pd

    # Load the saved model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(loadModel))
    else:
        model.load_state_dict(torch.load(loadModel,map_location=torch.device('cpu')))  # Load the saved parameters
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    features = testDataset[0]
    labels = testDataset[1]

    # Convert your input data to a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Use the model to make predictions
    with torch.no_grad():
        outputs = model(features)
        print(outputs)
        
        predictions = torch.where(outputs > 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Apply condition to outputs

    # Convert labels to binary
    binLabels = torch.where(labels != 0, torch.tensor(1.0), torch.tensor(0.0))

    _accuracy = (predictions == binLabels).float().mean()
    accuracy = "Accuracy: {:.6f}".format(_accuracy.item())
    
    if output == True:
        # Filter rows with predicted value 1, append the predicted value, and save to a new dataset
        filtered_rows = [row.tolist() + [pred.item()] for row, pred in zip(features, predictions) if pred == 1.0]
        df = pd.DataFrame(filtered_rows, columns=['Ux', 'Uy', 'Vx', 'omegaRES'])
        df.to_csv('non-zero_predictions.csv', index=False)

    return predictions, accuracy

def nnVisualize(train_losses,val_losses,val_accuracy):
    import matplotlib.pyplot as plt

    # Create a figure
    plt.figure()

    # Plot the training and validation losses for the model
    plt.plot(train_losses, '-', label='Training Loss')
    plt.plot(val_losses, '-', label='Validation Loss')
    #plt.plot(val_accuracy, '-o', label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()

def valLossComparasion():
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a figure
    plt.figure()

    minimal_val_losses = []
    for i in range(5):
        minimal_val_losses.append(nnTrain(visualize=False, save="XY-5"+str(i+1), splitDataset=DataPrep.split(*DataPrep.extract("dXY-5_1000.csv",limit=200)),model=nnArch(io=[2,1], hl=[16]), epochs=400, learningRate=0.0001))
    for i in range(5):
        minimal_val_losses.append(nnTrain(visualize=False, save="XY-5"+str(i+6), splitDataset=DataPrep.split(*DataPrep.extract("dXY-5_1000.csv",limit=200)),model=nnArch(io=[2,1], hl=[16]), epochs=400, learningRate=0.00001))

    # Plot the validation losses for all models
    plt.plot(minimal_val_losses, label='Minimal Validation Loss')

    plt.xlabel('Iteration index')
    plt.ylabel('Minimal Validation Loss')
    plt.legend()

    # Show the plot
    plt.show()


#modelArchitecture = nnArch(io=[9,1], hl=[220,160,128,96])
modelArchitecture = nnArch(io=[9,1], hl=[200,160,100,48])
#modelArchitecture = nnArch(io=[9,1], hl=[80,64,48])

path_to_dataset = "bin-dataset3D10k.csv"


# Trainig section
import copy
extractedData = DataPrep.extract(path=path_to_dataset,i=[1,9],o=[10,10],limit=0)
extractedDataCopy = copy.deepcopy(extractedData)
absmaxScaledData = DataPrep.scale(extractedDataCopy[0],extractedDataCopy[1],method="absmax",class_labels=True)

#nnTrain(save="classTest",splitDataset=DataPrep.split(*extractedData),model=modelArchitecture, epochs=50, learningRate=0.001, batch_size=32)
nnTrain(cli=False,visualize=False,save="classificator",splitDataset=DataPrep.split(*absmaxScaledData),model=modelArchitecture, epochs=10, learningRate=0.001, batch_size=32)


'''
# Predicting section
import copy
extractedData = DataPrep.extract(path="dataset3D10k.csv",i=[1,9],o=[10,10],limit=0)
extractedDataCopy = copy.deepcopy(extractedData)
absmaxScaledData = DataPrep.scale(extractedDataCopy[0],extractedDataCopy[1],method="absmax",class_labels=True)

#print(nnPredict(loadModel="classTest1_VL{2.982e-01}.pth", testDataset=extractedData,model=modelArchitecture,output=False))
predictions, accuracy = nnPredict(loadModel="classificator_50M_80_64_481_VL{7.147e-02}.pth", testDataset=absmaxScaledData,model=modelArchitecture,output=False)
print(DataPrep.inverseScale(extractedData[0],predictions,method="absmax"),accuracy)
'''