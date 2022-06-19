
# maybe vectorize
def q_loss(replay, model, discount): # replay is a ReplayBufferRecord
    Q = replay.reward + discount * model(replay.next)[replay.action]
    return (Q - model(replay.current))**2




