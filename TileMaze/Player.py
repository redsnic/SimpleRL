from abstract.Player import AbstractPlayer
from abstract.Game import MoveException

class Player(AbstractPlayer):

    def __init__(self):
        pass

    def setup(self, config): # config is a dict
        pass

    def act(self, game, replay_buffer=None):
        print('Insert next move: stay, up, down, left, right')
        while True:
            action = None
            x = input().split()
            if len(x) == 0:
                continue
            x = x[0].lower()
            if x == 'stay' or x=='.':
                action = 0
            elif x == 'up' or x=='u':
                action = 1
            elif x == 'down' or x=='d':
                action = 2
            elif x == 'left' or x=='l':
                action = 3
            elif x == 'right' or x=='r':
                action = 4
            if not action is None:
                try:
                    if replay_buffer:
                        replay_buffer.push(game, action)
                    game.update(action)
                    break
                except MoveException as s: # make a custom exception!
                    print('Invalid Move! Try Again')
             
