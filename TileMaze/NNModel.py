import torch
import torch.nn as nn
from NNBlocks.ResNet import ResNet
import numpy as np
 
class NNModel(torch.nn.Module):

    def __init__(self, board_size): # H, W
        super().__init__()
        self.h = board_size[0]
        self.w = board_size[1]

        self.resnet = ResNet( (None,2,self.h, self.w), depth=3, n_channels=32) # 5, 128
        inner_mlp_dim = 256
        self.mlp = nn.Sequential(
            nn.Linear(self.resnet.DRF_size+2, inner_mlp_dim),
            nn.BatchNorm1d(inner_mlp_dim),
            nn.ReLU(),
            nn.Linear(inner_mlp_dim, inner_mlp_dim),
            nn.BatchNorm1d(inner_mlp_dim),
            nn.ReLU(),
            nn.Linear(inner_mlp_dim, 5) # five possible actions
        )
        

    def forward(self, board, cursor): # board is BS:C:H:W, cursor is 2 el tensor
        DRF = self.resnet(board)
        DRF = torch.reshape(DRF, [DRF.shape[0], DRF.shape[1]]) 
        x = torch.cat([DRF, cursor], dim=1)
        x = self.mlp( x )

        return x

    def greedy_policy(self, board, cursor, valid_moves): # valid action must operate per batch
        all_the_bests = []
        options = self(board, cursor) 
        for i in range(options.shape[0]):
            best = -1
            best_score = -100000000
            for j in range(len(valid_moves[i])):
                if valid_moves[i][j] and options[i, j] > best_score:
                    best = j
                    best_score = options[i, j]
            all_the_bests.append(best)
        all_the_bests = torch.Tensor([all_the_bests])
        return all_the_bests
                

    def get_reward_horizon(self, board, cursor): # [action-reward-list]
        return np.sum(self(board, cursor))


    