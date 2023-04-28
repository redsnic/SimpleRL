from TileMaze.NNModel import NNModel
from DQN import ReplayBuffer
import torch
from tqdm import tqdm

def make_batch(replays, device):
    # input is ReplayBuffer (current, next, action, reward)

    current_board_batch = torch.stack( [ r.current[0] for r in replays ] ).to(device)
    current_cursor_batch = torch.stack( [ r.current[1] for r in replays ] ).to(device)

    next_board_batch = torch.stack( [ r.current[0] for r in replays ] ).to(device)
    next_cursor_batch = torch.stack( [ r.current[1] for r in replays ] ).to(device)

    actions = torch.Tensor([ r.action for r in replays ]).to(device)
    rewards = torch.Tensor([ r.reward for r in replays ]).to(device)
    future_rewards = torch.Tensor([ r.future_reward for r in replays ]).to(device)

    is_not_final = torch.Tensor([ r.is_not_final for r in replays ]).to(device)

    return {
        'current_boards' : current_board_batch,
        'current_cursors' : current_cursor_batch,
        'next_boards' : next_board_batch,
        'next_cursors' : next_cursor_batch,
        'actions' : actions,
        'rewards' : rewards,
        'is_not_final' : is_not_final,
        'future_rewards':future_rewards
    }



    


        


