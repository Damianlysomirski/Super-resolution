from torch import nn

class ESPCN(nn.Module):
     def __init__(self, scale_factor: int):
        super(ESPCN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3 * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),)

     def forward(self, x):
            x = self.model(x)
            return x