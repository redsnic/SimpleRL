from abstract.Game import AbstractGame, MoveException
import torch
import copy

class Game(AbstractGame):

    def __init__(self):
        super().__init__()
        self.cursor = (0,0)

    def initialize(self, repr = None): # repr is a dictionary, if given
        if repr:
            self._initialize_from_state(repr)
        else:
            self._base_initializiation()

    def _initialize_from_state(self, repr): # repr is a dictionary, if given
        self.board = repr['board']
        try:
            self.cursor = repr['cursor']
        except:
            self.cursor = (0,0)
        #try:
        #    self.time_penalty = repr['time_penalty']
        #except:
        #    self.time_penalty = 0
        
    def _base_initializiation(self):
        # keeps current board, return cursor to 0,0
        self.cursor = (0,0)

    def to_tensor(self): # torch.Tensor
        board = torch.split(torch.Tensor(self.board), 1, dim=2)
        cell_types = torch.reshape(board[0], board[0].shape[:2])
        cell_rewards = torch.reshape(board[1], board[1].shape[:2])
        return (torch.stack([cell_types, cell_rewards]), torch.Tensor(self.cursor))

    def simulate(self, action):
        new_state = copy.deepcopy(self)
        new_state.update(action)
        return new_state

    def update(self, action, verbose = False):
        # action encoding 0 Still, 1 Up, 2 Down, 3 Left, 4 Right
        new_cursor = None
        if action == 0:
            new_cursor = self.cursor
            raise('We should not get here...')
        elif action == 1:
            new_cursor = (self.cursor[0], self.cursor[1]-1)
        elif action == 2:
            new_cursor = (self.cursor[0], self.cursor[1]+1)
        elif action == 3:
            new_cursor = (self.cursor[0]-1, self.cursor[1])
        elif action == 4:
            new_cursor = (self.cursor[0]+1, self.cursor[1])
        if new_cursor is None:
            raise Exception('No action?')
        if new_cursor[0] < 0 or new_cursor[0] >= len(self.board[0]) or new_cursor[1] < 0 or new_cursor[1] >= len(self.board):
            raise MoveException('Invalid move! Current position:' + str(self.cursor) + '\n') 
        else:
            if self._at()[1] > 0: # (is_final, reward)
                self.board[self.cursor[1]][self.cursor[0]] = (self._at()[0], 0)
            if verbose:
                print(self)
            self.cursor = new_cursor

    def _at(self, pos=None):
        if pos:
            return self.board[pos[0]][pos[1]]
        return self.board[self.cursor[1]][self.cursor[0]]

    def status(self):
        if self._at()[0] == 1:
            return 'end'
        else:
            return 'normal'

    def get_current_reward(self): # out must be a float
        return self._at()[1]

    def __str__(self):
        s = 'Current state: \n'
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if i == self.cursor[1] and j == self.cursor[0]:
                    s+='*' #cursor
                elif self._at((i,j))[1] > 0:
                    s+='B' #Bonus
                elif self._at((i,j))[1] < 0:
                    s+='T' #Trap
                else:
                    s+='.'
            s+='\n'
        return s
    
    def show(self):
        print(self.__str__())

    def valid_moves(self):
        # still, up, down, left, right
        out = [True for i in range(5)]
        if self.cursor[0] <= 0:
            out[3] = False  
        if self.cursor[0] >= len(self.board[0])-1:
            out[4] = False
        if self.cursor[1] <= 0:
            out[1] = False
        if self.cursor[1] >= len(self.board)-1:
            out[2] = False
        # remove stay: (TODO)
        out[0] = False
        return out
        

