import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            #(Batchsize, channel, height, width) -> (Batchsize, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3,padding=1),

            #(Batchsize, 128, height, width) -> (Batchsize, 128, height, width)
            VAE_ResidualBlock(128, 128),

            #(Batchsize, 128, height, width) -> (Batchsize, 256, height, width)
            VAE_ResidualBlock(128, 128),

            #(Batchsize, 256, height, width) -> (Batchsize, 256, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),

            #(Batchsize, 256, height/2, width/2) -> (Batchsize, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            #(Batchsize, 256, height/2, width/2) -> (Batchsize, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            #(Batchsize, 256, height/2, width/2) -> (Batchsize, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),

            #(Batchsize, 256, height/4, width/4) -> (Batchsize, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            #(Batchsize, 512, height/4, width/4) -> (Batchsize, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/4, width/4) -> (Batchsize, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, padding=0, stride=2),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 512, height/8, width/8)
            nn.GroupNorm(32, 512),

            nn.SiLU(), 
            
            #(Batchsize, 512, height/8, width/8) -> (Batchsize, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1, stride=1),

            #(Batchsize, 8, height/8, width/8) -> (Batchsize, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0, stride=1),

        )


    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        # x: (Batchsize, channel, height, width)
        # noise: (Batchsize, channel, height, width)
        # return: (Batchsize, 8, height/8, width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        #(Batchsize, 8, height/8, width/8) -> 2 x (Batchsize, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, min=-30.0, max=20.0)

        # (Batchsize, 4, height/8, width/8) -> (Batchsize, 4, height/8, width/8)
        variance = torch.exp(log_variance)

        # (Batchsize, 4, height/8, width/8) -> (Batchsize, 4, height/8, width/8)
        stdev = torch.sqrt(variance)

        # N(0, 1) -> N(mean, variance)?
        x = mean + stdev * noise

        x *= 0.18215

        return x