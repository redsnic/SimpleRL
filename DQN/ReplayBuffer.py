import random

class ReplayBufferRecord():

    def __init__(self, state, action):
        self.current = state.to_tensor()
        self.action = action
        self.reward = state.reward()
        self.next = state.simulate(action).to_tensor()


# list based implementation
class ReplayBuffer():

    def __init__(self):
        self.stack = []

    def pop(self, N=1): # returns a list of size N of tuples 
        return random.choiches(self.stack, N)

    def push(self, state, action):
        record = ReplayBufferRecord(state, action)
        self.stack.append(record)
    
    # see if it is useful to add a tensor output utility
    
    


