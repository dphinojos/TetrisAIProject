from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import numpy as np
from problem import *
from node import *
from genetic_algorithm import *

class agent:
    pass

class Tetris(Problem):
    def __init__(self, initial, goal=None):
        super().__init__(initial, goal=None)

    def actions(self, state):
        allowed_actions = []

        #List of Actions into a numbered array
        for action in range(len(MOVEMENT)):
            allowed_actions.append(action)

        return allowed_actions
    
    def result(self, state, action):


        return action

    def value(self, state):
        
        pass



env = gym_tetris.make('TetrisA-v2')
env = JoypadSpace(env, MOVEMENT)

TetrisAI = Tetris(env)

TetrisPopulation = genetic_search(TetrisAI)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(0)
    env.render()

env.close()

