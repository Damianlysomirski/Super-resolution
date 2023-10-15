from torch import nn


class ESPCN(nn.Module):
    def __init__(self, scale_factor: int):
        """
        Initialize the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model.
        Parameters:
        - scale_factor (int): The scaling factor for the super-resolution task.
        """
        super(ESPCN, self).__init__()
        # Define the model as a sequence of layers
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3 * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        """
        Forward pass of the ESPCN model.
        Parameters:
        - x (torch.Tensor): Input image tensor.
        Returns:
        - x (torch.Tensor): Output super-resolved image tensor.
        """
        # Pass the input image through the model
        x = self.model(x)
        return x

