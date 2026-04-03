import numpy as np # for arrays

"""Environments"""

class BaseEnv():
    """Abstract Environment class"""
    def __init__(self):
        """Initialise environment's state, observation and action spaces"""
        pass
    
    def reset(self):
        """Reset the environment and agent's state"""
        pass

    def step(self, action):
        """Step the environment given an action """
        pass

    def render(self):
        """Render the environment graphically"""
        pass

"""GridWorlds - 2D grid environments for navigation"""

class GridWorld(BaseEnv):
    """
    GridWorld 1.0: Grid with obstacles and a single goal
    """
    def __init__(self, grid_size, start_pos, goal_pos, obstacles):
        """
        At initialisation, the script must pass the following initial conditions:
        - grid_size: 2D tuple of grid size e.g. (3,3) for 3x3 square, (2,5) for 2x5 rectangle
        - start_pos: 2D tuple of starting position on grid
        - goal_pos: 2D tuple of goal position on grid
        - obstacles: list of 2D tuples containing obstacle positions on grid
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles

        """
        Build the grid environment as a numpy array, and fill out the relevant cells
        - Agent location -> 1
        - Goal -> 2
        - Obstacles -> 3
        - Free cell -> 0
        """
        # initialise all cells as free (0s)
        self.grid = np.zeros(grid_size)

        self.grid[start_pos] = 1 #  starting pos
        self.grid[goal_pos] = 2 # goal pos
        for obstacle in obstacles:
            self.grid[obstacle] = 3

        """Define the observation space and action space"""
        
    def reset(self):
        """Reset function: reset the environment to its initial state using initial values"""
        # reset all cells to 0
        self.grid = np.zeros(grid_size)

        self.grid[start_pos] = 1 #  starting pos
        self.grid[goal_pos] = 2 # goal pos
        for obstacle in obstacles:
            self.grid[obstacle] = 3
    
    def step(self, action):
        """Apply an action to environment and return state, reward"""
        pass

if __name__ == "__main__":
    """Test grid environment"""
    grid_size = (3,3)

    start_pos = (0,0)
    goal_pos = (0,2)
    obstacles = [(0,1), (1,1)]

    grid_env = GridWorld(grid_size, start_pos, goal_pos, obstacles)

    print(grid_env.grid)