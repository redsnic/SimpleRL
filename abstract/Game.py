
class MoveException(Exception):
    pass


class AbstractGame():

    def __init__(self):
        pass

    def initialize(self, repr = None): # repr is a dictionary, if given
        if repr:
            self._initialize_from_state(repr)
        else:
            self._base_initializiation()

    def _initialize_from_state(self, repr): # repr is a dictionary, if given
        raise NotImplementedError()

    def _base_initializiation(self):
        raise NotImplementedError()

    def to_tensor(self): # torch.Tensor
        raise NotImplementedError()

    def simulate(self, action):
        new_state = self.copy()
        new_state.update(action)
        return new_state

    def update(self, action):
        raise NotImplementedError()

    def status(self):
        raise NotImplementedError()

    def get_current_reward(self): # out must be a float
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
    
    def show(self):
        raise NotImplementedError()

    def valid_moves(self):
        raise NotImplementedError()

