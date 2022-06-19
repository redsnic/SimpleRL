import random

class EpsilonGreedyPolicy():

    def __init__(self, initial_eps, rule : list): 
        self.eps = initial_eps
    
    def rule(self):
        raise NotImplementedError
        
    def decay(self, ratio):
        self.eps *= ratio

    def set_eps(self, eps):
        self.eps = eps
    
    def pick(self):
        if random.random() < self.eps:
            return random.choice(self.action_set)
        else:
            return self.rule()

    def __call__(self): # the out is an action of the action_set
        return self.pick()