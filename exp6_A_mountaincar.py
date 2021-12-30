import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.99

N_ACTIONS = 3
N_ATTRIBS = 2
ATTRIBUTES = [("Car Position", "continuous", -1, -1),
			  ("Car Velocity", "continuous", -1, -1)]

def eps_greedy(qtree, state, eps):
	leaf, action = qtree.predict(state)

	if np.random.random() < eps:
	# if np.random.random() < max([0.1, (5000 / i_episode)]):
		action = np.random.randint(0, N_ACTIONS)
	
	return leaf, action

def collect_data(qtree, n_episodes):
	env = gym.make("MountainCar-v0")

	for T in range(1, n_episodes):
		state = env.reset()
		action = 0
		reward = 0
		leaf = None
		done = False
		t = 0
		
		while not done:
			t += 1

			if leaf is None:
				leaf, action = qtree.predict(state)

			if np.random.random() < max([0.2, ((0.1 / 2) * n_episodes / T)]):
				action = np.random.randint(0, N_ACTIONS)
			
			next_state, reward, done, _ = env.step(action)
			next_leaf, next_action = qtree.predict(next_state)

			leaf.q_history[action].append((state, action, next_leaf.value, reward))
			leaf.state_history.append(state)

			leaf = next_leaf
			state = next_state
			action = next_action
	
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

def select_split(qtree, node, verbose=False):
	is_better_than = lambda score1, score2 : (score2[1] - score1[1] > 0.01) or (np.abs(score1[1] - score2[1]) < 0.01 and score1[0] > score2[0])

	if node.__class__.__name__ == "QLeaf":
		leaf = node
		if verbose:
			print(f"\n{'Left' if leaf.is_left else 'Right'} leaf{(' of x[' + str(leaf.parent.attribute) + '] <= ' + str(leaf.parent.value)) if leaf.parent is not None else ''}:")

		best_split = None
		best_score = [0, 1]

		for attribute_idx in range(N_ATTRIBS):
			attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
			
			cutoffs = []
			if attr_type == "continuous":
				if len(leaf.state_history) < 100:
					continue
				cutoffs = np.quantile([s[attribute_idx] for s in leaf.state_history], np.linspace(0.1, 0.9, 5))
			elif attr_type == "discrete":
				cutoffs = range(start_value, end_value)
			
			for cutoff in cutoffs:
				score = [0, 1]

				for action in range(N_ACTIONS):
					L_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] <= cutoff]
					R_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] > cutoff]

					if len(L_partition) > 0 and len(R_partition) > 0:
						kstest = stats.ks_2samp(L_partition, R_partition)
						score[0] += kstest[0]
						score[1] *= kstest[1]
						
				if verbose:
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
	env = gym.make("MountainCar-v0")

	for T in range(1, n_episodes):
		state = env.reset()
		action = None
		leaf = None
		reward = 0
		done = False
		
		while not done:
			if leaf is None:
				leaf, action = qtree.predict(state)

			if np.random.random() < 0.1:
				action = np.random.randint(0, N_ACTIONS)

			next_state, reward, done, _ = env.step(action)
			next_leaf, next_action = qtree.predict(next_state)
			
			if not done:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(next_leaf.q_values) - leaf.q_values[action])
			else:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])

			leaf.q_values[action] += delta_q
			# leaf.q_history[action].append(delta_q)

			state = next_state
			leaf = next_leaf
			action = next_action
	
	return qtree

def run_monte_carlo_control(qtree, n_episodes=1000):
	env = gym.make("MountainCar-v0")

	for T in range(1, n_episodes):
		episode = []
		state = env.reset()
		done = False

		while not done:
			leaf, action = qtree.predict(state)

			if np.random.random() < 0.5 * n_episodes / T:
				action = np.random.randint(0, N_ACTIONS)
			
			next_state, reward, done, _ = env.step(action)
			episode.append((leaf, action, reward))
			state = next_state

		G = 0
		for t in range(len(episode) - 1, -1, -1):
			leaf, action, reward = episode[t]
			
			G = DISCOUNT_FACTOR * G + reward

			has_appeared = False
			for i in range(0, t):
				leaf2, action2, _ = episode[i]
				if leaf == leaf2 and action == action2:
					has_appeared = True
					break
			
			if not has_appeared:
				leaf.q_history[action].append(G)
				leaf.q_values[action] = np.mean(leaf.q_history[action])
		
		qtree.reset_history()
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
	env = gym.make("MountainCar-v0")
	episode_rewards = np.zeros(n_episodes)

	for T in range(1, n_episodes):
		state = env.reset()
		reward = 0
		done = False
		t = 0
		
		while not done:
			t += 1
			_, action = qtree.predict(state)
			next_state, reward, done, _ = env.step(action)
			
			episode_rewards[T] += reward
			state = next_state
	
	return np.mean(episode_rewards)

# Initializing tree
qtree = QLeaf(parent=None, actions=["left", "right", "nothing"])
best_reward = 0
performance_history = []

for i in range(100):
	print(f"\n==> Iteration {i}:")
	# Data collecting phase
	qtree = collect_data(qtree, 1000)

	# Split phase
	qtree = update_datapoints(qtree)
	leaf, split, score = select_split(qtree, qtree, verbose=(i<10))
	if score[1] < 0.05:
		print(f">> Split ({ATTRIBUTES[split[0]][0]}, {split[1]}) is good enough! Score: (D: {score[0]}, p: {score[1]})")
		qtree = grow_tree(qtree, leaf, None, split)

		# Upkeep phase 
		print("\n> Running Q-Learning...")
		# qtree.reset_history()
		# qtree = run_monte_carlo_control(qtree, 20000)
		qtree = run_qlearning(qtree, 5000)
		qtree = update_value(qtree)

	qtree.print_tree()
	average_reward = get_average_reward(qtree, 100)
	performance_history.append(average_reward)
	print(f"Average reward for the tree is: {average_reward}")
	if average_reward > 1.05 * best_reward:
		best_reward = average_reward
		best_tree = copy.deepcopy(qtree)

	qtree.reset_history()

print(f"Best tree, with average reward {best_reward}:")
best_tree.print_tree()

with open('data/saved_tree_MC0', 'wb') as file:
  pickle.dump(best_tree, file)
  print("> Saved best tree to file 'data/saved_tree_MC0'!")

plt.plot(range(len(performance_history)), performance_history, color='blue')
plt.title("Performance history")
plt.xlabel("Tree size")
plt.ylabel("Average reward")
plt.show()