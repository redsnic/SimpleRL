from TileMaze.Game import Game
from TileMaze.NNBot import NNBot
from DQN.ReplayBuffer import ReplayBuffer
import torch
from tqdm import tqdm
import comet_ml

def make_board():
    return [
        [(0,0), (0,100), (0,0), (0,0)],
        [(1,-1000), (0,100), (0,0), (0,0)],
        [(0,0), (0,0), (0,100), (0,0)],
        [(0,0), (0,0), (1,-1000), (1,1000)]
    ] 

if __name__=='__main__':
    g = Game()
    board = make_board()
    g.initialize({'board':board})
    p = NNBot()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 0.01
    discount = 0.9
    init_eps=0.5

    p.setup(
        device=device, 
        init_eps=init_eps, 
        board_size=(len(board),len(board[0])),
        discount=discount,
        optimizer=torch.optim.Adam,
        learning_rate=lr
    )

    # comet.ml
    experiment = comet_ml.Experiment(api_key="vhIR3uyqsKyU4L7SA8fLCfTSC")
    experiment.log_parameter("training_rate", lr)
    experiment.log_parameter("discount_factor", discount)
    experiment.log_parameter("initial_eps", init_eps)

    

    n_iters = 300
    n_games = 100

    maximum_number_of_steps = 20

    for it in tqdm(range(n_iters)):
        buff = ReplayBuffer()
        for gms in range(n_games):
            p.set_eps(0.05+0.5*(1-it/n_iters))
            board = make_board()
            g.initialize({'board':board})
            i = 0
            while g.status() != 'end' and i < maximum_number_of_steps:
                p.model.eval() # batch norm requires bs>1
                p.act(g, replay_buffer=buff, verbose=False)
                i += 1
        p.model.train()
        p.train(buff, n_batchs=10, batch_size=128, epochs=3, device=device, comet=experiment)    
    
    print('Final gameplay choice:')
    
    board = make_board()
    g.initialize({'board':board})
    g.show()
    reward = 0
    i = 0
    while g.status() != 'end' and i < maximum_number_of_steps:
        p.model.eval()
        p.set_eps(0.)
        p.act(g, verbose=True)
        reward += g.get_current_reward()
        print('Reward: ' + str(reward))
        g.show()
        i += 1




    
    


