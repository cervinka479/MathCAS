import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_file, task='class'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract metrics
    epochs = range(1, len(df) + 1)
    train_losses = df['train_losses']
    val_losses = df['val_losses']
    
    if task == 'class':
        val_accuracies = df['val_accuracies']
    
    # Create a figure
    if task == 'class':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot training and validation losses
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Validation Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Losses')
        ax1.legend()
        
        # Plot validation accuracy
        ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        
        # Plot training and validation losses
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Validation Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Losses')
        ax1.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()

# Set the task
task = 'reg'

# Call the plot_metrics function with the specified CSV file and task
plot_metrics('NN_training_log/trial2.csv', task=task)