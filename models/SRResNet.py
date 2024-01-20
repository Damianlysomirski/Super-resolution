from torch import nn
import math
import torch

class ResBlock(nn.Module):
    def __init__(self, channel, kernel, pad):
        super().__init__()
        self.relu = nn.ReLU()
        self.inner_block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel, padding=pad)
        )

    def forward(self, img):
        x = self.relu(img)
        out = self.inner_block(x)
        return x + out

class PixelShuffleBlock(nn.Module):
    def __init__(self, channel, kernel, pad):
        super().__init__()
        upscale_factor = 2
        self.inner_block = nn.Sequential(
            nn.Conv2d(channel, channel*(upscale_factor**2), kernel, padding=pad), 
            nn.PixelShuffle(upscale_factor),
        )
    
    def forward(self, img):
        return self.inner_block(img)
    
class PixelShuffleBlockx3(nn.Module):
    def __init__(self, channel, kernel, pad):
        super().__init__()
        upscale_factor = 3
        self.inner_block = nn.Sequential(
            nn.Conv2d(channel, channel*(upscale_factor**2), kernel, padding=pad), 
            nn.PixelShuffle(upscale_factor),
        )
    
    def forward(self, img):
        return self.inner_block(img)

class SRResNet(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        channel = 64
        kernel = 3
        pad = 1
        
        self.conv = nn.Conv2d(3, channel, kernel, padding=pad)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channel, kernel, pad) for _ in range(9)],
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel, padding=pad),
        )
        if(scale_factor == 3):
            self.pixel_shuffle_blocks = nn.Sequential(
                PixelShuffleBlockx3(channel, kernel, pad),
                nn.Conv2d(channel, 3, kernel, padding=pad)
            )
            pass
        else:
            self.pixel_shuffle_blocks = nn.Sequential(
                *[PixelShuffleBlock(channel, kernel, pad) for _ in range(int( math.log2(scale_factor) ))],
                nn.Conv2d(channel, 3, kernel, padding=pad)
            )

    def forward(self, img):
        x = self.conv(img)
        out = self.res_blocks(x)
        out = self.pixel_shuffle_blocks(x+out).to(torch.float32)
        return out