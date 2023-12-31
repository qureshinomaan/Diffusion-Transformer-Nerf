import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention


def VAE_AttentionBlock(in_channels):
    def __init__(self, in_channels : int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(1, in_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x : (Batchsize, in_channels, height, width)
        residue = x

        n, c, h, w = x.shape
        
        #(Batchsize, in_channels, height, width) -> (Batchsize, in_channels, height*width)
        x = x.view(n, c, h*w)

        #(Batchsize, in_channels, height*width) -> (Batchsize, height*width, in_channels)
        x = x.transpose(-1, -2)

        #(Batchsize, height*width, in_channels) -> (Batchsize, height*width, in_channels)
        x = self.attention(x)
        
        #(Batchsize, height*width, in_channels) -> (Batchsize, in_channels, height*width)
        x = x.transpose(-1, -2)

        #(Batchsize, in_channels, height*width) -> (Batchsize, in_channels, height, width)
        x = x.view(n, c, h, w)

        x = x + residue

        return x



class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x : (Batchsize, in_channels, height, width)
        residue = x
        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        x = x + self.residual_layer(residue)

        return x
    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=1, stride=1),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/4, width/4)
            nn.UpSample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batchsize, 512, height/4, width/4) -> (Batchsize, 256, height/2, width/2)
            nn.UpSample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batchsize, 256, height/2, width/2) -> (Batchsize, 128, height, width)
            nn.UpSample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),


            nn.SiLU(),

            # (Batchsize, 128, height, width) -> (Batchsize, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1, stride=1),

        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x : (Batchsize, 4, height, width)
        x /= 0.18215

        for module in self:
            x = module(x)

        return x



