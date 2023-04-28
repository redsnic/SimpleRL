from TileMaze.Game import Game
from TileMaze.Player import Player

if __name__=='__main__':
    g = Game()
    board = [
        [(0,0), (0,100), (0,0), (0,0)],
        [(1,-1000), (0,100), (0,0), (0,0)],
        [(0,0), (0,0), (0,100), (0,0)],
        [(0,0), (0,0), (1,-1000), (1,1000)]
    ]   
    g.initialize({'board':board})
    p = Player()

    g.show()
    reward = 0
    while g.status() != 'end':
        p.act(g)
        reward += g.get_current_reward()
        print('Reward: ' + str(reward))
        g.show()

    
    

