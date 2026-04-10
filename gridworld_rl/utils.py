import random
import numpy as np

# define epsilon greedy policy
def epsilon_greedy(state, epsilon, Q_table, num_actions):
    if random.random() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Q_table[state[0]][state[1]])
    return action