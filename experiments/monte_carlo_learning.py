"""Implement Monte-Carlo learning with gridworld"""
import numpy as np
import matplotlib.pyplot as plt
import random

# import grid environment
from gridworld_rl.env import GridWorld
from gridworld_rl.utils import epsilon_greedy

#initialise environment
grid_size = (6, 8)

start_pos = (0, 0) # axis stars at top left (matrix index coordinates)
goal_pos = (5, 7)
# Create a maze-like pattern with obstacles
obstacles = [
    (4, 1), (4, 2), (4, 3),
    (3, 3), (3, 4), (3, 5),
    (2, 1), (2, 5), (2, 6),
    (1, 1), (1, 2), (1, 6),
]

env = GridWorld(grid_size, start_pos, goal_pos, obstacles)

# intialise q-table
Q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.num_actions))

"""Monte-carlo learning algorithm (every-visit)"""
num_eps = 5000
learning_rate = 0.1
epsilon = 1.0
discount = 0.99
total_rewards = [] # for tracking 

for i in range(num_eps):
    state = env.reset() # remember to reset environment!
    done = False
    episode_states = []
    episode_actions = []
    episode_rewards = []
    G = 0
    while not done:
        # record the current state
        episode_states.append(env.agent_pos)

        # epsilon greedy polcy
        action = epsilon_greedy(env.agent_pos, epsilon, Q_table, env.num_actions)

        # apply action to environment
        obs, reward, done, info = env.step(action)

        # append action and reward to episode
        episode_actions.append(action)
        episode_rewards.append(reward)

    # track rewards per episode
    total_rewards.append(sum(episode_rewards))
    # anneal epsilon per episode -> really important so to exploit more later (smaller epsilon)
    epsilon = max(0.01, epsilon * 0.995)

    # at the end of the episode, work backwards applying Bellman equation
    for j in reversed(range(len(episode_rewards))): # loop backwards
        G = episode_rewards[j] + discount * G

        # update Q table using bellman equation
        Q_table[episode_states[j][0]][episode_states[j][1]][episode_actions[j]] += (
            learning_rate * (G - Q_table[episode_states[j][0]][episode_states[j][1]][episode_actions[j]])
        )   
    
    print(f"Episode {i} complete | Episode Reward: {sum(episode_rewards)}")


# plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Plot 1: Environment scenario
env_grid = np.zeros(env.grid_size)
# Mark obstacles
for obs in obstacles:
    env_grid[obs[0]][obs[1]] = -1
# Mark goal
env_grid[goal_pos[0]][goal_pos[1]] = 2
# Mark start
env_grid[start_pos[0]][start_pos[1]] = 1

im0 = axes[0].imshow(env_grid, cmap='RdYlGn', vmin=-1, vmax=2, origin='upper')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')
axes[0].set_title('Environment Scenario')
# Add text annotations
for i in range(env.grid_size[0]):
    for j in range(env.grid_size[1]):
        if (i, j) == start_pos:
            axes[0].text(j, i, 'S', ha='center', va='center', fontweight='bold', color='black')
        elif (i, j) == goal_pos:
            axes[0].text(j, i, 'G', ha='center', va='center', fontweight='bold', color='white')
        elif (i, j) in obstacles:
            axes[0].text(j, i, 'X', ha='center', va='center', fontweight='bold', color='white')

# Plot 2: Total reward per episode
axes[1].plot(total_rewards, marker='o')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Total Reward')
axes[1].set_title('Total Reward per Episode')
axes[1].grid(True, alpha=0.3)

# Plot 3: Q-table value function heatmap (max over actions)
value_function = np.max(Q_table, axis=2)  # max over actions
im = axes[2].imshow(value_function, cmap='viridis', origin='upper')
axes[2].set_xlabel('Column')
axes[2].set_ylabel('Row')
axes[2].set_title('Value Function (Max Q-value per State)')
plt.colorbar(im, ax=axes[2], label='Value')

# Plot 4: Greedy policy arrows
axes[3].imshow(env_grid, cmap='RdYlGn', vmin=-1, vmax=2, origin='upper', alpha=0.3)
axes[3].set_xlabel('Column')
axes[3].set_ylabel('Row')
axes[3].set_title('Greedy Policy (Best Action per State)')

# Direction mappings: UP=0, DOWN=1, LEFT=2, RIGHT=3
# arrow(x, y, dx, dy) — dx is horizontal, dy is vertical
# origin='upper' means row 0 is at top, so UP = negative dy
directions = {
    0: (0, -0.3),   # UP: dx=0, dy=-0.3 (move up on screen)
    1: (0, 0.3),    # DOWN: dx=0, dy=+0.3 (move down on screen)
    2: (-0.3, 0),   # LEFT: dx=-0.3, dy=0 (move left on screen)
    3: (0.3, 0)     # RIGHT: dx=+0.3, dy=0 (move right on screen)
}

for i in range(env.grid_size[0]):
    for j in range(env.grid_size[1]):
        if (i, j) not in obstacles:
            greedy_action = np.argmax(Q_table[i, j, :])
            dx, dy = directions[greedy_action]   # unpack as dx, dy
            axes[3].arrow(j, i, dx, dy, head_width=0.15, head_length=0.1, 
                          fc='blue', ec='blue', alpha=0.7)
            
plt.tight_layout()
plt.show()
fig.savefig("experiments/results/monte_carlo.png")


    