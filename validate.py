import torch
from utils import psnr, ssim
from tqdm import tqdm

def validate(model, dataloader, optimizer, criterion, device):
    """
    Validate a neural network model.

    Parameters:
    - model (torch.nn.Module): The neural network model to be validated.
    - dataloader (torch.utils.data.DataLoader): Data loader for validation data.
    - optimizer (torch.optim.Optimizer): The optimizer (not used during validation).
    - criterion (torch.nn.Module): The loss function used for validation.
    - device (torch.device): The device (CPU or GPU) where validation takes place.

    Returns:
    - final_loss (float): Average loss over the entire validation dataset.
    - final_psnr (float): Average PSNR over the entire validation dataset.
    """
    # Set the model in evaluation mode (no gradient computation)
    model.eval()
    running_loss = 0.0  # Initialize running loss
    running_psnr = 0.0  # Initialize running PSNR (Peak Signal-to-Noise Ratio)
    running_ssim = 0.0

    # Use torch.no_grad() to disable gradient tracking for validation
    with torch.no_grad():
        # Iterate over the validation data loader
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract the input image and labels from the data
            image_data = data[0].to(device)
            label = data[1].to(device)

            # Perform validation using automatic mixed precision (AMP) for better performance
            with torch.cuda.amp.autocast():
                outputs = model(image_data)
                loss = criterion(outputs, label)

            # Accumulate the loss for each item (total items in a batch = batch size)
            running_loss += loss.item()

            # Calculate PSNR (Peak Signal-to-Noise Ratio) for the batch
            batch_psnr = psnr(label, outputs)
            batch_ssim = ssim(label, outputs)
            running_psnr += batch_psnr
            running_ssim += batch_ssim

    # Calculate the final loss as the average over the entire validation dataset
    final_loss = running_loss / len(dataloader.dataset)

    # Calculate the final PSNR as the average over the entire validation dataset
    final_psnr = running_psnr / len(dataloader) 

    # Calculate the final PSNR as the average over the entire validation dataset
    final_ssim = running_ssim / len(dataloader)

    return final_loss, final_psnr, final_ssim