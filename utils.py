import math
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

def ssim(label, outputs):
    """
    For now use this
    """
    label = label.cpu().detach()
    outputs = outputs.cpu().detach()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(outputs, label)

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (PSNR), a measure of image quality.
    
    Parameters:
    - label (Tensor): Ground truth values (e.g., target image).
    - outputs (Tensor): Predicted values (e.g., generated image).
    - max_val (float): Maximum possible pixel value (default is 1.0 for normalized images).

    Returns:
    - PSNR (float): Peak Signal to Noise Ratio value.
    
    PSNR = 20 * log10(max_val) - 10 * log10(MSE)
    
    Reference:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    
    Note: Both the label and outputs should be normalized if max_val is 1.
    """
    # Convert label and outputs tensors to numpy arrays
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    
    # Calculate the pixel-wise difference between outputs and label
    diff = outputs - label
    
    # Calculate the root mean squared error (RMSE)
    rmse = math.sqrt(np.mean(diff ** 2))
    
    # Handle the case where RMSE is zero to avoid division by zero
    if rmse == 0:
        return 100.0
    else:
        # Calculate PSNR using the formula
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
    
"""
For now its not working
"""
# def ssim(label, outputs):
#     """
#     Calculate the Structural Similarity Index (SSIM) between two images using PyTorch tensors.

#     Parameters:
#     - label (torch.Tensor): Ground truth image tensor.
#     - outputs (torch.Tensor): Predicted image tensor.

#     Returns:
#     - ssim_score (float): SSIM score (between -1 and 1).
#     """
#     # Ensure tensors are on the CPU and in float64 format
#     label = label.squeeze().cpu().detach()
#     outputs = outputs.squeeze().cpu().detach()

#     # Constants for SSIM calculation
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     # Calculate mean of label and outputs across height and width (dimensions 2 and 3)
#     mu_label = torch.mean(label, dim=0, keepdim=True)
#     mu_outputs = torch.mean(outputs, dim=0, keepdim=True)

#     # Calculate variance of label and outputs across height and width (dimensions 2 and 3)
#     var_label = torch.var(label, dim=0, keepdim=True)
#     var_outputs = torch.var(outputs, dim=0, keepdim=True)

#     # Calculate covariance between label and outputs across height and width (dimensions 2 and 3)
#     covar = torch.mean((label - mu_label) * (outputs - mu_outputs), dim=0, keepdim=True)

#     # SSIM formula components
#     numerator = (2 * mu_label * mu_outputs + C1) * (2 * covar + C2)
#     denominator = (mu_label ** 2 + mu_outputs ** 2 + C1) * (var_label + var_outputs + C2)

#     # Calculate SSIM score
#     ssim_score = torch.mean(numerator / denominator)

#     return ssim_score


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
    plt.savefig("./results/psnr_plot")


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
    plt.savefig("./results/loss_plot")
