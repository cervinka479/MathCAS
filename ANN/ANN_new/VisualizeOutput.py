import ANN
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))

epochs = 200
#plt.plot(range(1, epochs + 1), ANN.HiddenLayer1(hidden_size = 100, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='g')
#plt.plot(range(1, epochs + 1), ANN.HiddenLayer1(hidden_size = 1000, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='y')

epochs = 200
plt.plot(range(1, epochs + 1), ANN.HiddenLayer2(hidden_size1 = 128, hidden_size2 = 128, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='y')
plt.plot(range(1, epochs + 1), ANN.HiddenLayer2(hidden_size1 = 256, hidden_size2 = 256, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='g')
#plt.plot(range(1, epochs + 1), ANN.HiddenLayer2(hidden_size1 = 1000, hidden_size2 = 1000, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='y')

epochs = 200
plt.plot(range(1, epochs + 1), ANN.HiddenLayer3(hidden_size1 = 256, hidden_size2 = 256, hidden_size3 = 256, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='r')
plt.plot(range(1, epochs + 1), ANN.HiddenLayer3(hidden_size1 = 512, hidden_size2 = 256, hidden_size3 = 128, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='b')

#plt.plot(range(1, epochs + 1), ANN.HiddenLayer5(hidden_size1 = 1000, hidden_size2 = 1000, hidden_size3 = 1000, hidden_size4 = 1000, hidden_size5 = 1000, learning_rate = 0.001, num_epochs = epochs), marker='o', linestyle='-', color='y')

plt.title('Loss (MSE) vs. Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()