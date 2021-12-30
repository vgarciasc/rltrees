import gym
import copy
import pdb
import numpy as np
from numpy.lib.function_base import average
import mdp
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.99

N_ACTIONS = 2
N_ATTRIBS = 1
ATTRIBUTES = [("i", "discrete", 0, 4)]

def eps_greedy(qtree, state, eps):
	leaf, action = qtree.predict(state)

	if np.random.random() < eps:
	# if np.random.random() < max([0.1, (5000 / i_episode)]):
		action = np.random.randint(0, N_ACTIONS)
	
	return leaf, action

def collect_data(qtree, n_episodes):
	for T in range(1, n_episodes):
		state = 0
		reward = 0
		done = False
		t = 0
		
		while not done:
			t += 1
			leaf, action = eps_greedy(qtree, [state], max([0.2, ((0.1 / 2) * n_episodes / T)]))
			next_state, reward, done = MDP(state, action, t)
			next_leaf, _ = qtree.predict([next_state])

			leaf.q_history[action].append(([state], action, next_leaf.value, reward))

			state = next_state
	
	return qtree

def update_datapoints(node):
	if node.__class__.__name__ == "QLeaf":
		leaf = node
		
		for action_id in range(N_ACTIONS):
			for (s, a, v, r) in leaf.q_history[action_id]:
				q = r + DISCOUNT_FACTOR * v
				leaf.full_q_history[a].append((s, a, q))
	else:
		if node.left is not None:
			update_datapoints(node.left)
		if node.right is not None:
			update_datapoints(node.right)

	return node

def select_split(qtree, node):
	is_better_than = lambda score1, score2 : (score2[1] - score1[1] > 0.01) or (np.abs(score1[1] - score2[1]) < 0.01 and score1[0] > score2[0])

	if node.__class__.__name__ == "QLeaf":
		leaf = node
		print(f"\n{'Left' if leaf.is_left else 'Right'} leaf{(' of x[' + str(leaf.parent.attribute) + '] <= ' + str(leaf.parent.value)) if leaf.parent is not None else ''}:")

		best_split = None
		best_score = (0, 1)

		for attribute_idx in range(N_ATTRIBS):
			attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
			
			for cutoff in range(start_value, end_value):
				score = [0, 1]

				for action in range(N_ACTIONS):
					L_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] <= cutoff]
					R_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] > cutoff]

					if len(L_partition) > 0 and len(R_partition) > 0:
						kstest = stats.ks_2samp(L_partition, R_partition)
						score[0] += kstest[0]
						score[1] *= kstest[1]
						
				print(f"> Split {(attr_name, cutoff)} has score (D: {score[0]}, p-value: {score[1]})")

				if is_better_than(score, best_score):
					best_split = (attribute_idx, cutoff)
					best_score = score
			
		return leaf, best_split, best_score
	else:
		if node.left is not None:
			left_leaf, left_best_split, left_best_score = select_split(qtree, node.left)
		if node.right is not None:
			right_leaf, right_best_split, right_best_score = select_split(qtree, node.right)

		if is_better_than(left_best_score, right_best_score):
			return left_leaf, left_best_split, left_best_score
		else:
			return right_leaf, right_best_split, right_best_score

def run_qlearning(qtree, n_episodes):
	state = 0
	reward = 0
	done = False

	for T in range(1, n_episodes):
		state = 0
		reward = 0
		done = False
		t = 0
		
		while not done:
			t += 1

			leaf, action = qtree.predict([state])
			if np.random.random() < 0.1:
				action = np.random.randint(0, N_ACTIONS)

			next_state, reward, done = MDP(state, action, t)
			next_leaf, next_action = qtree.predict([next_state])
			
			if not done:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(next_leaf.q_values) - leaf.q_values[action])
			else:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])

			leaf.q_values[action] += delta_q
			leaf.q_history[action].append(delta_q)

			state = next_state
	
	return qtree

def update_value(node):
	if node.__class__.__name__ == "QLeaf":
		leaf = node
		leaf.value = np.max(leaf.q_values)
	else:
		if node.left is not None:
			update_value(node.left)
		if node.right is not None:
			update_value(node.right)

	return node

def get_average_reward(qtree, n_episodes):
	episode_rewards = np.zeros(n_episodes)

	for T in range(1, n_episodes):
		state = 0
		reward = 0
		done = False
		t = 0
		
		while not done:
			t += 1
			_, action = qtree.predict([state])
			next_state, reward, done = MDP(state, action, t)
			
			episode_rewards[T] += reward
			state = next_state
	
	return np.mean(episode_rewards)

# Setting up environment
MDP = mdp.mdp4_step

# Initializing tree
qtree = QLeaf(parent=None, actions=["left", "right"])
best_reward = 0

for i in range(5):
	print(f"\n==> Iteration {i}:")
	# Data collecting phase
	qtree = collect_data(qtree, 10000)

	# Split phase
	qtree = update_datapoints(qtree)
	leaf, split, score = select_split(qtree, qtree)
	if score[1] < 0.05:
		print(f">> Split {split} is good enough!")
		qtree = grow_tree(qtree, leaf, None, split)

	# Upkeep phase 
	# qtree.reset_history()
	# qtree = run_monte_carlo_control(qtree, 50000)
	print("\n> Running Q-Learning...")
	qtree = run_qlearning(qtree, 5000)

	qtree = update_value(qtree)

	qtree.print_tree()
	average_reward = get_average_reward(qtree, 100)
	print(f"Average reward for the tree is: {average_reward}")
	if average_reward > 1.05 * best_reward:
		best_reward = average_reward
		best_tree = copy.deepcopy(qtree)

	qtree.reset_all()

print(f"Final tree, with average reward {average_reward}:")
best_tree.print_tree()