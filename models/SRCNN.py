from torch import nn
import torch

class SRCNN(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__() 
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.Conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.Conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.Conv3 = nn.Conv2d(32, 3, 5, 1, 2)
        self.Relu = nn.ReLU()
        
    def forward(self, x):
        out = self.upsample(x)
        out = self.Relu(self.Conv1(out))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)
        out = torch.clip(out, 0.0, 1.0)
        return out