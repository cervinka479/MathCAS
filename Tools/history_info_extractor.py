import re
import matplotlib.pyplot as plt

def plot_loss_and_accuracy(paths):
    # Create a figure with six subplots: two rows and three columns
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, path in enumerate(paths):
        # Initialize lists to store the data
        training_loss = []
        validation_loss = []
        validation_accuracy = []

        # Open the file and read the data
        with open(path, 'r') as file:
            for line in file:
                # Use regular expressions to extract the values
                match = re.search(r'Training Loss: (.*), Validation Loss: (.*), Validation Accuracy: (.*)', line)
                if match:
                    training_loss.append(float(match.group(1)))
                    validation_loss.append(float(match.group(2)))
                    validation_accuracy.append(float(match.group(3)))

        # Plot the training and validation loss on the first row
        axs[0, i].plot(training_loss, label='Training Loss')
        axs[0, i].plot(validation_loss, label='Validation Loss')
        axs[0, i].set_title(f'Losses for {path}')
        axs[0, i].legend()

        # Plot the validation accuracy on the second row
        axs[1, i].plot(validation_accuracy, label='Validation Accuracy', color='green')
        axs[1, i].set_title(f'Accuracy for {path}')
        axs[1, i].legend()

    # Adjust the layout
    plt.tight_layout()
    # Show the plot
    plt.show()

plot_loss_and_accuracy(['history_10M_32_24_32.txt', 'history_10M_32_24_3200.txt','history_50M_32_24_3200.txt'])
