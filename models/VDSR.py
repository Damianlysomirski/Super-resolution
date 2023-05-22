from torch import nn

class VDSR(nn.Module):
    def __init__(self, scale_factor: int):
      super().__init__()
      layers = []

      #Upsample
      self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

      #Input layer
      layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))

      #Resiudal layers
      for i in range(18):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())

      #Output layer
      layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1))

      self.model = nn.Sequential(*layers)

    def forward(self, x):
      x = self.upsample(x)
      out = self.model(x) + x
      return out