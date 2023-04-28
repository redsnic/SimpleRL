
# maybe vectorize
#def q_loss(replay, model, discount): # replay is a ReplayBufferRecord
#    Q = replay.reward + discount * torch.argmax(model(replay.next), dim=1)
#    return (Q - model(replay.current)[replay.action])**2




