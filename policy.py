"""Policies to test on GridWorld"""
from env import GridWorld
import numpy as np

class RandomPolicy():
    """Random policy - pick actions from uniform distribution over possible discrete actions"""
    def __init__(self, environment):
        """Input the environment to extract the action space size"""
        self.num_actions = environment.num_actions # actions are assumed to enumerate 0 -> N

    def act(self, obs):
        """Act on random policy (observation input but not used)"""
        return np.random.choice(self.num_actions) # pick random from np.arange(num_actions)
    
if __name__ == '__main__':

    grid_size = (4,5)

    start_pos = (2,2) # axis stars at top right (matrix index coordinates)
    goal_pos = (0,2)
    obstacles = [(0,1), (1,1)]

    grid_env = GridWorld(grid_size, start_pos, goal_pos, obstacles)
    policy = RandomPolicy(grid_env)

    action = policy.act(grid_env._get_obs())
    print(grid_env.step(action))
