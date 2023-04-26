from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

import cupy as np

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

alpha = 0.9  # learning rate
gamma = 0.4  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 1000

num_states = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
num_actions = env.action_space.n 

q_table = np.zeros((num_states, num_actions))

def flatten_state(state):
    # Convert 3D state to 1D array
    state = state.ravel()
    
    return state

for episode in range(num_episodes):
    state = env.reset()
    state = flatten_state(state)
 
    done = False
    t = 0
    score = 0
    total_reward = 0

    while not done:
        # Choose action epsilon-greedily
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state, :]))
        
        # Take action and observe next state and reward
        next_state, reward, done, info = env.step(action)
        next_state = flatten_state(next_state)

        #env.render()
        # Update Q-value
        if np.max(q_table[next_state, :]) == 0:
            #Error checking
            q_table = np.vstack((q_table, np.zeros((1, num_actions))))
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        # Update state
        state = next_state
        t += 1
        total_reward += reward
        score = info['score']
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Print progress
    print(f"Episode {episode}: {t} timesteps, Epsilon: {epsilon}, Score: {score}")
    print(f"Total Reward: {total_reward}")

np.save('q_table.npy', q_table)
np.save('epsilon.npy', epsilon)

env.close()
