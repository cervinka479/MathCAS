import re
import matplotlib.pyplot as plt

def class_extractor(paths):
    # Create a figure with six subplots: two rows and three columns
    fig, axs = plt.subplots(2, len(paths), figsize=(5*len(paths), 10))

    for i, path in enumerate(paths):
        # Initialize lists to store the data
        training_loss = []
        validation_loss = []
        validation_accuracy = []

        # Open the file and read the data
        with open(path, 'r') as file:
            for line in file:
                # Use regular expressions to extract the values
                
                match = re.search(r'Training Loss: (.*), Validation Loss: (.*), Validation Accuracy: (.*), Estimated Remaining Time:', line)
                
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

def reg_extractor(path):
    # Initialize lists
    training_loss = []
    validation_loss = []

    # Open the file and read the data
    with open(path, 'r') as file:
        for line in file:
            # Use regular expressions to extract the values
            match = re.search(r'Training Loss: (.*), Validation Loss: (.*),', line)
            
            if match:
                training_loss.append(float(match.group(1)))
                validation_loss.append(float(match.group(2)))

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training Loss and Validation Loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#class_extractor(['class_deeper.txt','class_double.txt'])
    
reg_extractor('regression_10M_80_64_48.txt')