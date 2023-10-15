from torch import nn

class Bicubic(nn.Module):
    def __init__(self, scale_factor: int):
        """
        Initialize the Bicubic Upsampling model.
        Parameters:
        - scale_factor (int): The scaling factor for bicubic upsampling.
        """
        super().__init__()
        # Upsample layer using bicubic interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
        """
        Forward pass of the Bicubic Upsampling model.
        Parameters:
        - x (torch.Tensor): Input image tensor.
        Returns:
        - out (torch.Tensor): Output upsampled image tensor using bicubic interpolation.
        """
        # Upsample the input image using bicubic interpolation
        out = self.upsample(x)
        return out