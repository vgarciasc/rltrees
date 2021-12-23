import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from control_tree import CTNode, CTLeaf

N_ACTIONS = 2
N_ATTRIBS = 1
ATTRIBUTES = [("i", "discrete", 0, 4)]

def mdp_step(state, action):
	next_state, reward = state, 0
	done = False

	if (state == 0 or state == 1):
		next_state = 2
	if (state == 2):
		next_state = 1 if action == 0 else 0
	
	rewards = [[-1.0, -1.3, -0.7], [+0.7, +1.0, -0.5]]
	reward = rewards[action][state]

	# if state == 4 and action == 1:
	# 	done = True
	# if state == 0 and action == 0:
	# 	done = True

	
	return next_state, reward, done

learning_rate = 0.1
discount_factor = 0.9

# q_values = [[0, 0], [0, 0], [0, 0]]
q_values = [[0, 0], [0, 0]]
total_rewards = []

# get_leaf = lambda s : s
get_leaf = lambda s : 0 if s in [0, 1] else 1

state = 0
reward, total_reward = 0, 0
done = False

for _ in range(0, 100000):
	leaf = get_leaf(state)
	action = np.argmax(q_values[leaf])

	if np.random.random() < 0.1:
		action = np.random.choice([0, 1])

	next_state, reward, done = mdp_step(state, action)
	next_leaf = get_leaf(next_state)
	
	if not done:
		delta_q = learning_rate * (reward + discount_factor * np.max(q_values[next_leaf]) - q_values[leaf][action])
	else:
		delta_q = learning_rate * (reward + discount_factor * 0 - q_values[leaf][action])
	
	q_values[leaf][action] += delta_q
	state = next_state

print(f"Q values: \n{q_values[0][1]}, {q_values[0][0]}\n{q_values[1][1]}, {q_values[1][0]}")
# print(f"Q values: \n{q_values[0][1]}, {q_values[0][0]}\n{q_values[1][1]}, {q_values[1][0]}\n{q_values[2][1]}, {q_values[2][0]}")