from torch import nn

class Bicubic(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__() 
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
      out = self.upsample(x)
      return out