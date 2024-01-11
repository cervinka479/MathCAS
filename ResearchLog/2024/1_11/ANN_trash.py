import DataPrep

def nnArch(io=[3,1], hl=[10, 10]):
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

def nnTrain(splitDataset=[["train_input"],["train_output"],["val_input"],["val_output"],["test_input"],["test_output"]], model=nnArch(),optimizer="adam",learningRate=0.01,criterion="mse",batch_size=4, epochs=50, save="test", visualize=True, cli=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import copy

    # Unpack your dataset
    train_input, train_output, val_input, val_output, test_input, test_output = splitDataset

    # Convert your dataset to PyTorch tensors
    train_data = TensorDataset(torch.tensor(train_input, dtype=torch.float32), torch.tensor(train_output, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(val_input, dtype=torch.float32), torch.tensor(val_output, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(test_input, dtype=torch.float32), torch.tensor(test_output, dtype=torch.float32))
    
    # Create a DataLoader for your dataset
    train_loader = DataLoader(train_data, batch_size)
    val_loader = DataLoader(val_data, batch_size)
    test_loader = DataLoader(test_data, batch_size)

    print(model)

    match optimizer:
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), learningRate)
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), learningRate)

    match criterion:
        case "mse":
            criterion=torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), save + '.pth')

def nnPredict(loadModel, testDataset, model=nnArch()):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    
    # Load the saved model
    model.load_state_dict(torch.load(loadModel))  # Load the saved parameters
    model.eval()  # Set the model to evaluation mode

    features = testDataset[0]
    labels = testDataset[1]

    # Convert your input data to a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Preprocess the labels for binary classification
    labels = (labels != 0).float()

    # Use the model to make predictions
    with torch.no_grad():
        outputs = model(features)
        # Convert outputs into class predictions
        #predictions = (outputs > 0.5).float()
        predictions = outputs

    # Compute the accuracy
    correct = (predictions == labels).float().sum()
    accuracy = correct / len(labels)
    print("Accuracy: {:.3f}".format(accuracy.item()))

    return predictions, accuracy


modelArchitecture = nnArch(io=[3,1], hl=[10,10])

'''# Trainig section
extractedData = DataPrep.extract(path="dOmegaRES10k.csv",i=[1,3],o=[4,4],limit=80000)
froScaledData = DataPrep.scale(extractedData[0],extractedData[1],method="fro")

nnTrain(save="classTest",splitDataset=DataPrep.split(*extractedData),model=modelArchitecture, epochs=100, learningRate=0.001, batch_size=8)
#nnTrain(save="classTest",splitDataset=DataPrep.split(*froScaledData),model=modelArchitecture, epochs=50, learningRate=0.001, batch_size=8)'''

# Predicting section
extractedData = DataPrep.extract(path="dTest.csv",i=[1,3],o=[4,4])
froScaledData = DataPrep.scale(extractedData[0],extractedData[1],method="fro")

print(nnPredict(loadModel="classTest.pth", testDataset=extractedData,model=modelArchitecture)[0])
#print(DataPrep.inverseScale(extractedData[0],nnPredict(loadModel="8kTestModel1_VL{2.297e-06}.pth", testDataset=froScaledData,model=modelArchitecture)[0],method="fro"))