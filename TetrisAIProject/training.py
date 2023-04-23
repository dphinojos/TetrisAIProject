from QNetwork import *
from logger import *

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

#New Network
tetris = TetrisAI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

#Loading In case of crash
tetris.load_checkpoint()

logger = MetricLogger(save_dir)

episodes = 2000 - 640
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = tetris.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)
        env.render()

        # Remember
        tetris.cache(state, next_state, action, reward, done)
        # Learn
        q, loss = tetris.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=tetris.exploration_rate, step=tetris.current_step)

tetris.save()