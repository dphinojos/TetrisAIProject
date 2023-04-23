from QNetwork import *

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

tetris = TetrisAI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=None)
tetris.load()

games = 10

for g in range(games):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = tetris.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)
        env.render()

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break