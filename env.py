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
        Define the observation space and action space
        - Observation space is information the agent sees
        - Action space is the possible actions an agent can take
        In v1: 
        Observation space: (agent_pos, goal_pos) 
                            -> agent know where it is, and where the goal is (same as state-space)
        Action space: (UP, DOWN, LEFT, RIGHT)
        """
        # initialise observation to starting state (in this case obs = state space as it sees goal too)
        self.obs = np.array([start_pos, goal_pos])

        # initialise action space dict - useful for debugging what actions its taking
        self.action_space = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }

    def reset(self):
        """
        Reset function: reset the environment to its initial state using initial values
        RETURNS: intial observation (starting pos and goal)
        """
        
        self.obs = np.array([start_pos, goal_pos])
        return self.obs
    
    def step(self, action):
        """
        Apply an action to environment and return state, reward
        ACTION: 0 - UP, 1 - DOWN, 2 - LEFT, 3 - RIGHT

        1. Apply action
        2. Handle if you hit a wall or obstacle
        3. Compute reward using reward function
        4. Check if done
        5. Return observation', reward
        """

        # copy current position of agent
        new_pos = self.obs[0]

        # intialise done flag
        done = False

        # intialise obstacle hit flag
        hit_obstacle = False

        """Apply action to get new position of agent"""
        if action == 0:
            # UP
            new_pos[0] -= 1
        elif action == 1:
            # DOWN
            new_pos[0] += 1
        elif action == 2:
            # LEFT
            new_pos[1] -= 1
        elif action == 3:
            # RIGHT
            new_pos[1] += 1

        """Handle hitting the walls or an obstacle"""
        # check if new pos puts you outside of the grid
        if (new_pos[0] >= self.grid_size[0] or new_pos[0] < 0
            or new_pos[1] >= self.grid_size[1] or new_pos[1] < 0):
            # stay where you are
            new_pos = self.obs[0]
        # check if hit an obstacle (for now this is a soft hit that keeps you where you are)
        elif tuple(new_pos) in self.obstacles:
            new_pos = self.obs[0]
            hit_obstacle = True
        
        """
        Compute rewards for this action
        For v1 use simple reward function:
            if goal reached = +1
            if obstacle hit = -0.1
            if not reached goal yet = -0.01 (to penalise long paths)
            if wall hit = -0.01
        """
        reward = 0
        if (new_pos == self.goal_pos).all():
            reward = 1
            done = True
        elif hit_obstacle:
            reward = -0.1
        else:
            reward = -0.01

        # update observation
        self.obs[0] = new_pos

        """Return observation, reward and some extra flags"""
        return  self.obs, reward, done, hit_obstacle

    def render(self):
        pass

if __name__ == "__main__":
    """Test grid environment"""
    grid_size = (3,3)

    start_pos = (0,0) # axis stars at top right (matrix index coordinates)
    goal_pos = (0,2)
    obstacles = [(0,1), (1,1)]

    grid_env = GridWorld(grid_size, start_pos, goal_pos, obstacles)

    print(grid_env.step(3))