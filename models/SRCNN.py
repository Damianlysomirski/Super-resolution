from torch import nn
import torch

class SRCNN(nn.Module):
    def __init__(self, scale_factor: int):
        """
        Initialize the SRCNN (Super-Resolution Convolutional Neural Network) model.
        based on:   https://arxiv.org/pdf/1501.00092v3.pdf
        Parameters:
        - scale_factor (int): The scaling factor for the super-resolution task.
        """
        super().__init__()
        # Upsample layer using bicubic interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        # Convolutional layers
        self.Conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.Conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)
        # ReLU activation
        self.Relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the SRCNN model.
        Parameters:
        - x (torch.Tensor): Input image tensor.
        Returns:
        - out (torch.Tensor): Output super-resolved image tensor.
        """
        # Upsample the input image
        out = self.upsample(x)
        # Apply Conv1, ReLU, Conv2, ReLU, Conv3 sequentially
        out = self.Relu(self.Conv1(out))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out).to(torch.float32)
        return out
    
