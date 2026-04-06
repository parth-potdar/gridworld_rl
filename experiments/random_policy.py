"""Test the grid environment using a random policy"""
from gridworld_rl.env import GridWorld
from gridworld_rl.policy import RandomPolicy

import numpy as np
import matplotlib.pyplot as plt

"""Initilise environment and policy"""
grid_size = (20,20)

start_pos = (2,2) # axis stars at top left (matrix index coordinates)
goal_pos = (0,2)
obstacles = [(0,1), (1,1)]

grid_env = GridWorld(grid_size, start_pos, goal_pos, obstacles)
policy = RandomPolicy(grid_env)

obs = grid_env.reset() # reset environment

done = False
while not done: # until goal reached
    action = policy.act(obs) # take action
    obs, reward, done, info = grid_env.step(action)

    grid_env.render()