import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
from qtree import QNode, QLeaf, grow_tree, save_tree

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99

N_ACTIONS = 4
N_ATTRIBS = 8
ATTRIBUTES = [("X Position", "continuous", -1, -1),
			  ("Y Position", "continuous", -1, -1),
			  ("X Velocity", "continuous", -1, -1),
			  ("Y Velocity", "continuous", -1, -1),
			  ("Angle", "continuous", -1, -1),
			  ("Angular Velocity", "continuous", -1, -1),
			  ("Leg 1 is Touching", "categorical", [0, 1], -1),
			  ("Leg 2 is Touching", "categorical", [0, 1], -1)]

def eps_greedy(qtree, state, eps):
	leaf, action = qtree.predict(state)

	if np.random.random() < eps:
	# if np.random.random() < max([0.1, (5000 / i_episode)]):
		action = np.random.randint(0, N_ACTIONS)
	
	return leaf, action

def collect_data(qtree, n_episodes):
	env = gym.make("LunarLander-v2")

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

			if done:
				reward = 0

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
			elif attr_type == "categorical":
				cutoffs = start_value
			
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
	env = gym.make("LunarLander-v2")

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
				reward = 0
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])

			leaf.q_values[action] += delta_q
			# leaf.q_history[action].append(delta_q)

			state = next_state
			leaf = next_leaf
			action = next_action
	
	return qtree

def run_monte_carlo_control(qtree, n_episodes=1000):
	env = gym.make("LunarLander-v2")

	for T in range(1, n_episodes):
		episode = []
		state = env.reset()
		done = False

		while not done:
			leaf, action = qtree.predict(state)

			if np.random.random() < 0.5 * n_episodes / T:
				action = np.random.randint(0, N_ACTIONS)
			
			next_state, reward, done, _ = env.step(action)
			if done:
				reward = 0
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
	env = gym.make("LunarLander-v2")
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

def run_CUT(qtree, cut_iters=100, collect_data_iters=1000, q_learning_iters=5000, average_reward_iters=100, verbose=False):
	best_reward = -99999
	reward_history = []
	no_improve = 0

	for i in range(cut_iters):
		print(f"\n==> Iteration {i}, tree size {qtree.get_size()}:")
		# Data collecting phase
		qtree = collect_data(qtree, collect_data_iters)

		# Split phase
		qtree = update_datapoints(qtree)
		leaf, split, score = select_split(qtree, qtree, verbose=False)
		if score[1] < 0.05:
			print(f">> Split ({ATTRIBUTES[split[0]][0]}, {split[1]}) is good enough! Score: (D: {score[0]}, p: {score[1]})")
			qtree = grow_tree(qtree, leaf, None, split)
		else:
			no_improve += 1
			if no_improve == 3:
				break

		# Upkeep phase 
		print("\n> Running Q-Learning...")
		# qtree.reset_history()
		# qtree = run_monte_carlo_control(qtree, 20000)
		qtree = run_qlearning(qtree, q_learning_iters)
		qtree = update_value(qtree)

		if verbose:
			qtree.print_tree()
		average_reward = get_average_reward(qtree, average_reward_iters)
		reward_history.append((qtree.get_size(), average_reward))
		print(f"Average reward for the tree is: {average_reward}")
		if average_reward > 1.05 * best_reward:
			best_reward = average_reward
			best_tree = copy.deepcopy(qtree)

		qtree.reset_history()

	print(f"Best tree, with average reward {best_reward} and size {best_tree.get_size()}:")
	if verbose:
		best_tree.print_tree()
	return best_tree, reward_history

def merge_leaves_reward(qtree, node, n_trials=100):
	history = []

	if node.__class__.__name__ == "QLeaf":
		return qtree, []
	
	if node.left.__class__.__name__ == "QNode":
		qtree, new_history = merge_leaves_reward(qtree, node.left)
		history += new_history
	if node.right.__class__.__name__ == "QNode":
		qtree, new_history = merge_leaves_reward(qtree, node.right)
		history += new_history
	
	if node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QLeaf":
		print("")
		average_reward = get_average_reward(qtree, n_trials)
		print(f"The average reward of the tree is {average_reward}.")

		was_left = None
		merged_q_values = [node.left.q_values, node.right.q_values][np.argmax([np.max(node.left.q_values), np.max(node.right.q_values)])]
		merged_leaf = QLeaf(node.parent, node.parent is not None and node == node.parent.left, node.left.actions, merged_q_values)
		if node.parent is not None:
			if node == node.parent.left:
				was_left = True
				node.parent.left = merged_leaf
			else:
				was_left = False
				node.parent.right = merged_leaf
		else:
			qtree = merged_leaf
		print(f"Merged leaves of node '{ATTRIBUTES[node.attribute][0]} <= {node.value}'!")
		# if node.parent is not None:
		# 	print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
		
		new_average_reward = get_average_reward(qtree, n_trials)
		print(f"Got average reward {new_average_reward} after merge.")

		if new_average_reward < (0.9 if average_reward > 0 else 1.1) * average_reward:
			print("Average reward was reduced too much. Undoing merge...")
			if node.parent is not None:
				if was_left == True:
					node.parent.left = node
				else:
					node.parent.right = node
			else:
				qtree = node
		
			new_average_reward = get_average_reward(qtree, n_trials)
			print(f"Undid merge, got average reward {new_average_reward}.")
			# if node.parent is not None:
			# 	print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
		else:
			print(f"Appending {(qtree.get_size(), new_average_reward)}")
			history.append((qtree.get_size(), new_average_reward))

	if (node.left.__class__.__name__ == "QNode" and node.right.__class__.__name__ == "QLeaf") or \
		(node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QNode"):
		print("")
		average_reward = get_average_reward(qtree, n_trials)
		print(f"The average reward of the tree is {average_reward}.")

		was_left = None
		if node.left.__class__.__name__ == "QNode":
			node.left.parent = node.parent
		else:
			node.right.parent = node.parent

		if node.parent is not None:
			if node == node.parent.left:
				was_left = True
				node.parent.left = node.left if node.left.__class__.__name__ == "QNode" else node.right
			else:
				was_left = False
				node.parent.right = node.left if node.left.__class__.__name__ == "QNode" else node.right
		else:
			qtree = node.left if node.left.__class__.__name__ == "QNode" else node.right
		child_node = node.left if node.left.__class__.__name__ == "QNode" else node.right
		print(f"Routed from node '{ATTRIBUTES[node.attribute][0]} <= {node.value}' to its subtree '{ATTRIBUTES[child_node.attribute][0]} <= {child_node.value}'!")
		# if node.parent is not None:
		# 	print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
		# 	print(f"Now '{ATTRIBUTES[(child_node).attribute][0]} <= {(child_node).value}' has parent '{ATTRIBUTES[(child_node).parent.attribute][0]} <= {(child_node).parent.value}'")

		# pdb.set_trace()

		new_average_reward = get_average_reward(qtree, n_trials)
		print(f"Got average reward {new_average_reward} after merge.")

		if new_average_reward < (0.9 if average_reward > 0 else 1.1) * average_reward:
			(node.left if node.left.__class__.__name__ == "QNode" else node.right).parent = node
			print("Average reward was reduced too much. Undoing merge...")

			if node.parent is not None:
				if was_left == True:
					node.parent.left = node
				else:
					node.parent.right = node
			else:
				qtree = node
		
			new_average_reward = get_average_reward(qtree, n_trials)
			print(f"Undid subtree routing, got average reward {new_average_reward}.")
			if node.parent is not None:
				print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
				print(f"Now '{ATTRIBUTES[child_node.attribute][0]} <= {child_node.value}' has parent '{ATTRIBUTES[child_node.parent.attribute][0]} <= {child_node.parent.value}'")
		else:
			print(f"Appending {(qtree.get_size(), new_average_reward)}")
			history.append((qtree.get_size(), new_average_reward))

	return qtree, history

# Initializing tree
qtree = QLeaf(parent=None, actions=["nop", "left engine", "main engine", "right engine"])

history = []

for i in range(10):
	print(f"Phase {i}")
	qtree, reward_history = run_CUT(qtree, 10, 100, 100, 100, verbose=True)
	save_tree(qtree, "_growing", get_average_reward(qtree, 10))
	history.append(reward_history)

	# qtree = run_qlearning(qtree, 10000)

	reward_history = []
	new_history = []
	k = 0
	while len(reward_history) == 0 or len(new_history) != 0:
		qtree, new_history = merge_leaves_reward(qtree, qtree)
		reward_history += new_history
		save_tree(qtree, "_pruning", get_average_reward(qtree, 10))

		k += 1
		if k > 5:
			break
	history.append(reward_history)

current_x = 0
xticks = [[0], [1]]
for (i, phase) in enumerate(history):
	color = "blue"
	if i % 2 == 0:
		color = "red"

	plt.plot(range(current_x, current_x + len(phase)), [b for (a, b) in phase], color=color)
	xticks[0].append(current_x + len(phase) - 1)
	xticks[1].append(phase[len(phase) - 1][0])
	current_x += len(phase)
xticks[0].append(current_x)
plt.xticks(xticks[0], xticks[1])
plt.title("Performance history")
plt.xlabel("Tree size")
plt.ylabel("Average reward")
plt.show()
pdb.set_trace()