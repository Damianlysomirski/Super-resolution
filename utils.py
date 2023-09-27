import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
    
def check_device():
    print(torch.cuda.is_available())
    print("\n")
    print(torch.cuda.current_device())
    print("\n")
    print(torch.cuda.device(0))
    print("\n")
    print(torch.cuda.device_count())
    print("\n")
    print(torch.cuda.get_device_name(0))

# def plot_psnr(train_psnr, val_psnr, model_save_name):
#     # PSNR plots.
#     plt.figure(figsize=(10, 7))
#     plt.plot(train_psnr, color='green', label='train PSNR dB')
#     plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
#     plt.xlabel('Epochs')
#     plt.ylabel('PSNR (dB)')
#     plt.legend()
#     plt.savefig("./plots_new/psnr_" + model_save_name +".png")
#     plt.close()

# def plot_loss(train_loss, val_loss, model_save_name):
#     # Loss plots.
#     plt.figure(figsize=(10, 7))
#     plt.plot(train_loss, color='orange', label='train loss')
#     plt.plot(val_loss, color='red', label='validataion loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig("./plots_new/loss_" + model_save_name +".png")
#     plt.close()


def plot_psnr_new(train_psnr, val_psnr, name):
    """
    Create a plot to visualize training and validation PSNR (Peak Signal-to-Noise Ratio) values over epochs.

    Parameters:
    - train_psnr (list): List of training PSNR values for each epoch.
    - val_psnr (dict): Dictionary containing validation PSNR values with epochs as keys.
    - name (str): Title of the plot.

    """
    # Create a new figure for the plot with a specific size
    plt.figure(figsize=(10, 7))
    
    # Plot the training PSNR using an orange line and label it as 'train PSNR'
    plt.plot(train_psnr, color='orange', label='train PSNR')
    
    # Extract the x (epochs) and y (PSNR) values from the validation PSNR dictionary
    x, y = zip(*val_psnr.items())
    
    # Plot the validation PSNR using a red line and label it as 'validation PSNR'
    plt.plot(x, y, color='red', label='validation PSNR')
    
    # Label the x-axis as 'Epochs'
    plt.xlabel('Epochs')
    
    # Label the y-axis as 'PSNR'
    plt.ylabel('PSNR')
    
    # Add a legend to distinguish between training and validation PSNR lines
    plt.legend()
    
    # Set the title of the plot with "PSNR" prefix and the provided 'name' argument
    plt.title("PSNR " + str(name))
    
    # Display the plot
    plt.show()

# def plot_psnr_new(train_psnr, val_psnr, name):
#     plt.figure(figsize=(10, 7))
#     plt.plot(train_psnr, color='orange', label='train loss')
#     x, y = zip(*val_psnr.items())
#     plt.plot(x, y, color='red', label='validataion loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(name)
#     plt.show()

def plot_loss_new(train_loss, val_loss, name):
    """
    Create a plot to visualize training and validation loss over epochs.

    Parameters:
    - train_loss (list): List of training loss values for each epoch.
    - val_loss (dict): Dictionary containing validation loss values with epochs as keys.
    - name (str): Title of the plot.

    """
    # Create a new figure for the plot with a specific size
    plt.figure(figsize=(10, 7))

    # Plot the training loss using an orange line and label it as 'train loss'
    plt.plot(train_loss, color='orange', label='train loss')
    
    # Extract the x (epochs) and y (loss) values from the validation loss dictionary
    x, y = zip(*val_loss.items())
    
    # Plot the validation loss using a red line and label it as 'validation loss'
    plt.plot(x, y, color='red', label='validation loss')
    
    # Label the x-axis as 'Epochs'
    plt.xlabel('Epochs')
    
    # Label the y-axis as 'Loss'
    plt.ylabel('Loss')
    
    # Add a legend to distinguish between training and validation loss lines
    plt.legend()

    # Set the title of the plot with "LOSS" prefix and the provided 'name' argument
    plt.title("LOSS " + str(name))

    # Display the plot
    plt.show()
