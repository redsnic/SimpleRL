

class AbstractPlayer():

    def __init__(self):
        raise NotImplementedError

    def setup(self, **config): # config is a dict
        raise NotImplementedError

    def act(self, game):
        raise NotImplementedError
