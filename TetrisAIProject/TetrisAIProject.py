from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np
from problem import *
from node import *
from genetic_algorithm import *

class Tetris(Problem):

    def __init__(self, initial, goal=None):
        super().__init__(initial, goal=None)

    def actions(self, state):
        allowed_actions = []

        #List of Actions into a numbered array
        for action in range(len(SIMPLE_MOVEMENT)):
            allowed_actions.append(action)

        return allowed_actions
    
    def result(self, state, action):
        global reward

        state, reward, _, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}")
        #env.reset()

        return state

    def value(self, state):
        #Fitness Function
        global reward
        
        return reward



env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
TetrisAI = Tetris(state)

#TetrisGenetics = genetic_search(TetrisAI)

done = True
total_reward = 0
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(0)
    total_reward += reward
    print(f"Reward: {total_reward}")
    env.render()

env.close()

