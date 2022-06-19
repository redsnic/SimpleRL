import torch

class AbstractNNModel(torch.nn.Model):

    def __init__(self):
        super().__init__()

    def forward(self, *argparams, **kwparams):
        return NotImplementedError

    def greedy_policy(self, *argparams, **kwparams):
        return NotImplementedError

    def get_reward_horizon(self, game): # [action-reward-list]
        return NotImplementedError # this might be implemented automatically

    