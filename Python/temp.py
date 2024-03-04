import re
import matplotlib.pyplot as plt

# Initialize lists to store the data
training_loss = []
validation_loss = []
validation_accuracy = []

# Open the file and read the data
with open('convergence_history.txt', 'r') as file:
    for line in file:
        # Use regular expressions to extract the values
        match = re.search(r'Training Loss: (.*), Validation Loss: (.*), Validation Accuracy: (.*)', line)
        if match:
            training_loss.append(float(match.group(1)))
            validation_loss.append(float(match.group(2)))
            validation_accuracy.append(float(match.group(3)))

# Create a figure with two subplots: one for losses and one for accuracy
fig, axs = plt.subplots(2)

# Plot the training and validation loss on the first subplot
axs[0].plot(training_loss, label='Training Loss')
axs[0].plot(validation_loss, label='Validation Loss')
axs[0].legend()

# Plot the validation accuracy on the second subplot
axs[1].plot(validation_accuracy, label='Validation Accuracy', color='green')
axs[1].legend()

# Show the plot
plt.show()