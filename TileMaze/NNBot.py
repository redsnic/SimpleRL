from DQN.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from TileMaze.NNModel import NNModel
from abstract.Player import AbstractPlayer
from abstract.Game import MoveException
from TileMaze.training import make_batch
from tqdm import tqdm   
import torch
import comet_ml

class NNBot(AbstractPlayer):

    def __init__(self): # H,W
        pass
    
    def setup(self, device='cpu', init_eps=0.1, board_size=(10,10), discount=0.9, optimizer=None, learning_rate=0.0001): # init_eps, board_size, device
        self.pol = EpsilonGreedyPolicy(init_eps)
        self.model = NNModel(board_size).to(device)
        self.pol.rule = lambda xs, valid : self.model.greedy_policy(torch.stack([x[0] for x in xs]).to(device), torch.stack([x[1] for x in xs]).to(device), valid)
        self.device = device
        self.discount = discount
        self.optimizer = optimizer(self.model.parameters(), learning_rate)

    def set_eps(self, eps):
        self.pol.eps = eps

    def act(self, game, replay_buffer=None, verbose=True):

        action = self.pol([game])[0] # single batch

        if verbose:
            if action == 0:
                print('Action: stay')
            if action == 1:
                print('Action: up')
            if action == 2:
                print('Action: down')
            if action == 3:
                print('Action: left')
            if action == 4:
                print('Action: right')
                
        try:
            if replay_buffer:
                replay_buffer.push(game, action)
            game.update(action, verbose)
        except MoveException as s: # make a custom exception!
            print(s)
            raise Exception('Bot has chosen an invalid action! -> ' + str(action))

    def train(self, replay_buffer, n_batchs, batch_size, epochs, device, comet=None):

        for block in range(n_batchs):

            batch_group_loss = 0

            batch = make_batch(replay_buffer.pop(N=batch_size), self.device)

            for epoch in range(1,epochs+1):

                self.optimizer.zero_grad()

                # Q loss (old, new are the older future state of the board)
                old_outs = self.model(batch['current_boards'], batch['current_cursors'])
                new_outs = self.model(batch['next_boards'], batch['next_cursors'])
                action_values = torch.take(old_outs, torch.arange(0,batch['actions'].shape[0]).to(device)*old_outs.shape[1] + batch['actions'].long())
                future_best = torch.max(new_outs, dim=1)
                loss = torch.square(
                    self.discount*future_best.values*batch['is_not_final'] + (1-batch['is_not_final'])*self.discount*batch['future_rewards'] + batch['rewards'] - action_values 
                    )

                loss.sum().backward()

                if not comet is None:
                    batch_loss = loss.sum()
                    batch_group_loss += batch_loss
                    comet.log_metric('batch_loss', batch_loss, step=None, epoch=None, include_context=True)
                    comet.log_metric('mean_batch_loss', batch_loss/batch_size, step=None, epoch=None, include_context=True)

                self.optimizer.step()

        if not comet is None:
            comet.log_metric('batch_group_loss', batch_group_loss, step=None, epoch=None, include_context=True)

                
        


                    
             
