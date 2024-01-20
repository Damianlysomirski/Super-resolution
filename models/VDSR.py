from torch import nn

class VDSR(nn.Module):
    def __init__(self, scale_factor: int):
        """
        Initialize the VDSR (Very Deep Super-Resolution) neural network model.
        Parameters:
        - scale_factor (int): The scaling factor for the super-resolution task.
        """
        super().__init__()
        layers = []
        
        # Upsample layer using bicubic interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        
        # Input layer
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.ReLU())

        # Residual layers
        for i in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False))
        # Create the neural network model using the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the VDSR model.
        Parameters:
        - x (torch.Tensor): Input image tensor.
        Returns:
        - out (torch.Tensor): Output super-resolved image tensor.
        """
        # Upsample the input image
        x = self.upsample(x)
        # Pass the upsampled image through the model and add it to the original input
        out = self.model(x) + x
        return out