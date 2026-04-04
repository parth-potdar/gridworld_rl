import numpy as np # for arrays
import matplotlib.pyplot as plt # for rendering grid

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
        self.goal_pos = np.array(goal_pos)
        self.obstacles = set(obstacles) # cast to set for O(1) lookup

        """
        Define the observation space and action space
        - Observation space is information the agent sees
        - Action space is the possible actions an agent can take
        In v1: 
        State space: agent_pos and goal pos
        Observation space: (agent_pos, goal_pos) 
                            -> agent know where it is, and where the goal is (same as state-space)
        Action space: (UP, DOWN, LEFT, RIGHT)
        """
        # intialise the agent's position
        self.agent_pos = np.array(start_pos)

        # initialise action space dict - useful for debugging what actions its taking
        self.action_space = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
    
    def _get_obs(self):
        """Helper function to create the observation separately to state"""
        return np.concatenate([self.agent_pos, self.goal_pos]) # return observation 2D array

    def reset(self):
        """
        Reset function: reset the environment to its initial state using initial values
        RETURNS: intial observation
        """
        
       # intialise the agent's position back to start
        self.agent_pos = np.array(start_pos)
        return self._get_obs()
    
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
        new_pos = self.agent_pos.copy()

        # intialise done flag
        done = False

        hit_obstacle = False
        info = {} # initalise info dictionary for other flags

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
            new_pos = self.agent_pos.copy()
        # check if hit an obstacle (for now this is a soft hit that keeps you where you are)
        elif tuple(new_pos) in self.obstacles:
            new_pos = self.agent_pos.copy()
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

        # update agents position
        self.agent_pos = new_pos

        # add hit obstacle flag to info
        info['hit_obstacle'] = hit_obstacle

        """Return observation, reward and some extra flags"""
        return  self._get_obs(), reward, done, info

    def render(self):
        """
        Plot the environment using matplotlib:
        - agent's current location - blue
        - agent's starting location - light blue
        - goal - green
        - grid and boundary
        - obstacles - red
        """

        fig, ax = plt.subplots()
        rows, cols = self.grid_size

        # Draw grid
        for x in range(cols + 1):
            ax.axvline(x, color='gray', linewidth=1)
        for y in range(rows + 1):
            ax.axhline(y, color='gray', linewidth=1)

        # Draw boundary
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)

        # Draw start
        # Have to flip the tuples as Rectangle expects x (horizontal col) and y (vertical row)
        # but we are using matrix indexing which is opposite
        ax.add_patch(plt.Rectangle(self.start_pos[::-1], 1, 1, color='lightblue'))
        # Draw agent
        ax.add_patch(plt.Rectangle(self.agent_pos[::-1], 1, 1, color='blue'))
        # Draw goal
        ax.add_patch(plt.Rectangle(self.goal_pos[::-1], 1, 1, color='green'))
        # Draw obstacles
        for obs in self.obstacles:
            ax.add_patch(plt.Rectangle(obs[::-1], 1, 1, color='red'))

        # Invert y-axis so (0,0) is at bottom-left
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.show()
    

if __name__ == "__main__":
    """Test grid environment"""
    grid_size = (4,5)

    start_pos = (2,2) # axis stars at top right (matrix index coordinates)
    goal_pos = (0,2)
    obstacles = [(0,1), (1,1)]

    grid_env = GridWorld(grid_size, start_pos, goal_pos, obstacles)

    
