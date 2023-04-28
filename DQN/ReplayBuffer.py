import random

class ReplayBufferRecord():

    def __init__(self, state, action):
        self.current = state.to_tensor()
        self.action = action
        self.reward = state.get_current_reward()
        next_state = state.simulate(action)
        self.next = next_state.to_tensor()
        self.future_reward = next_state.get_current_reward()
        self.is_not_final = float(next_state.status() != 'end')


# list based implementation
class ReplayBuffer():

    def __init__(self):
        self.stack = []

    def pop(self, N=1): # returns a list of size N of tuples 
        return random.choices(self.stack, k=N)

    def push(self, state, action):
        record = ReplayBufferRecord(state, action)
        self.stack.append(record)
    
    # see if it is useful to add a tensor output utility
    
    


