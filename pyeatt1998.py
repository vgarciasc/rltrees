import gym
import math
import numpy as np
import time
import pdb

from numpy.core.defchararray import split
from control_tree import CTNode, CTLeaf

# Implementation of Pyeatt and Howe's (1998) decision tree for reinforcement learning

N_ACTIONS = 2
N_STATE = 4
HISTORY_MIN_SIZE = 100

def split_information_gain(data):
	classes = range(N_ACTIONS)
	log2 = lambda x : 0 if x == 0 else math.log2(x)

	best_ig = 0
	best_split = None

	for attribute in range(N_STATE):
		data.sort(key = lambda x : x[0][attribute])
		values = [x[attribute] for (x, _, _) in data]

		for i in range(len(values) - 1):
			value = values[i] + (values[i+1] - values[i]) / 2

			p_classes = [len([y for (_, y, _) in data if y == c]) / len(data) for c in classes]
			entropy = np.sum([ - p_classes[c] * log2(p_classes[c]) for c in classes])

			left_data = [x for (x, y, _) in data if x[attribute] <= value]
			left_labels = [y for (x, y, _) in data if x[attribute] <= value]
			p_left = [len([y for y in left_labels if y == c]) / len(left_labels) for c in classes]
			entropy_left = np.sum([ - p_left[c] * log2(p_left[c]) for c in classes])
			proportion_left = (len(left_data) / len(data))

			right_data = [x for (x, y, _) in data if x[attribute] > value]
			right_labels = [y for (x, y, _) in data if x[attribute] > value]
			p_right = [len([y for y in right_labels if y == c]) / len(right_labels) for c in classes]
			entropy_right = np.sum([ - p_right[c] * log2(p_right[c]) for c in classes])
			proportion_right = (len(right_data) / len(data))

			information_gain = entropy - proportion_left * entropy_left - proportion_right * entropy_right

			if information_gain > best_ig:
				best_ig = information_gain
				best_split = (attribute, value)
	
	return best_ig, best_split

def split_sse_fast(data):
	best_sse = math.inf
	best_split = None
	N = len(data)

	for attribute in range(N_STATE):
		data.sort(key = lambda x : x[0][attribute])
		values = [x[attribute] for (x, _, _) in data]

		left_data = []
		right_data = [q for (x, _, q) in data]

		for i in range(1, len(values) - 1):
			left_data.append(right_data[0])
			right_data.pop(0)

			var_left = np.var(left_data)
			var_right = np.var(right_data)

			sse = var_left * i + var_right * (N - i)

			if sse < best_sse:
				best_sse = sse
				value = values[i-1] + (values[i] - values[i-1]) / 2
				best_split = (attribute, value)
	
	return best_sse, best_split

def split_variance_fast(data):
	best_vg = 0
	best_split = None
	N = len(data)

	for attribute in range(N_STATE):
		data.sort(key = lambda x : x[0][attribute])
		var_data = np.var([q for (_, _, q) in data])
		values = [x[attribute] for (x, _, _) in data]

		left_data = []
		right_data = [q for (x, _, q) in data]

		for i in range(1, len(values) - 1):
			left_data.append(right_data[0])
			right_data.pop(0)

			proportion_left = i / N
			proportion_right = 1 - i / N
			var_left = np.var(left_data)
			var_right = np.var(right_data)

			variance_gain = var_data - var_left*proportion_left - var_right*proportion_right

			if variance_gain > best_vg:
				best_vg = variance_gain
				value = values[i-1] + (values[i] - values[i-1]) / 2
				best_split = (attribute, value)
	
	return best_vg, best_split

class PHCTLeaf(CTLeaf):
	def __init__(self, parent, is_left):
		self.parent = parent
		self.is_left = is_left
		self.history = []
		self.q_values = np.zeros(N_ACTIONS)

	def print_tree(self, level=1):
		print(" " * 2 * level, str(np.argmax(self.q_values)))
	
	def predict(self, state):
		return self, np.argmax(self.q_values)

env = gym.make("CartPole-v1")

eps = 0.1
learning_rate = 0.8
discount_factor = 0.9
ctree = PHCTLeaf(None, None)

total_rewards = []

for i_episode in range(10000):
	state = env.reset()
	reward = 0
	done = False

	total_reward = 0

	while not done:
		# env.render()
		was_random_move = False
		
		leaf, action = ctree.predict(state)
		
		if np.random.random() < eps:
			action = np.random.randint(0, env.action_space.n)
			was_random_move = True

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = ctree.predict(next_state)
		delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])

		if not was_random_move:
			leaf.q_values[action] += delta_q

		leaf.history.append([state, action, delta_q])

		if len(leaf.history) > HISTORY_MIN_SIZE:
			# q_values = [q for (_, label, q) in leaf.history if label == action]
			q_values = [q for (_, label, q) in leaf.history]

			mean = np.mean(q_values) # TODO: pensar mais sobre isso
			std_dev = np.std(q_values)

			print(f"|mean| < 2 * stddev --> {'{:.3f}'.format(mean)} < 2 * {'{:.3f}'.format(std_dev)} --> {'{:.3f}'.format(mean)} < {'{:.3f}'.format(2 * std_dev)}")

			if np.abs(mean) < 2 * std_dev:
				# history = [(data, label, q) for (data, label, q) in leaf.history if label == action]
				history = [(data, label, q) for (data, label, q) in leaf.history]
				gain, best_split = split_sse_fast(history)

				if best_split is None:
					print("No split found.")
				else:
					print(f"Average total reward for previous tree: {np.mean(total_rewards)}")
					total_rewards = []

					print(f"split selected: (x[{best_split[0]}] <= {best_split[1]}), variance gain: {gain}")

					new_node = CTNode(best_split, None, None)
					new_node.left = PHCTLeaf(new_node, True)
					new_node.right = PHCTLeaf(new_node, False)
					
					if leaf.parent is None:
						ctree = new_node
					else:
						if leaf.is_left:
							leaf.parent.left = new_node
						else:
							leaf.parent.right = new_node
					
				# ctree.print_tree()
				print("-" * 25)

		total_reward += reward
		state = next_state

	# print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
	total_rewards.append(total_reward)
		
env.close()

print("Average reward per episode:", np.mean(total_rewards))