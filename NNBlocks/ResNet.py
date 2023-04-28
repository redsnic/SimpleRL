import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, input_shape, depth=50, kernel_size=3, n_channels=2048): # input_shape: N:C:H:W
        super().__init__()
        in_channels = input_shape[1]
        #in_w = input_shape[2]
        #in_h = input_shape[3]
        self.DRF_size = n_channels

        self.res_blocks = nn.ModuleList()

        for i in range(depth):
            if i > 0:
                self.res_blocks.append(nn.Sequential(
                        nn.Conv2d(n_channels, n_channels, kernel_size, padding='same'),
                        nn.BatchNorm2d(n_channels),
                        nn.ReLU(),
                        nn.Conv2d(n_channels, n_channels, kernel_size, padding='same'),
                        nn.BatchNorm2d(n_channels)
                    )
                )
            else: # first case
                self.res_blocks.append(nn.Sequential(
                        nn.Conv2d(in_channels, n_channels, kernel_size, padding='same'),
                        nn.BatchNorm2d(n_channels),
                        nn.ReLU()
                    )
                )

        self.pool = nn.AdaptiveAvgPool2d(1) # extract DRF


    def forward(self, x):

        is_first_layer = True
        for l in self.res_blocks:
            if is_first_layer:
                if isinstance(x, torch.Tensor):   # FIX THIS 
                    x = l(x)
                else:
                    x = l(torch.stack(x))
                is_first_layer = False
            else:
                x = l(x) + x
        
        x = self.pool(x)
        
        return x
