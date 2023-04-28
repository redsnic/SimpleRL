import random
import torch

class EpsilonGreedyPolicy():

    def __init__(self, initial_eps : list): 
        self.eps = initial_eps
    
    def rule(self):
        raise NotImplementedError
        
    def decay(self, ratio):
        self.eps *= ratio

    def set_eps(self, eps):
        self.eps = eps
    
    def pick(self, games):
        if random.random() < self.eps:
            moves_per_game = [ g.valid_moves() for g in games ]
            # the action set must be computed from the game
            return [random.choice([i for i in range(len(moves)) if moves[i]]) for moves in moves_per_game]
        else:
            board = [ g.to_tensor() for g in games ]
            return self.rule(board, [ g.valid_moves() for g in games ])

    def __call__(self, game): # the out is an action of the action_set
        return self.pick(game)