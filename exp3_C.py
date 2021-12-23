import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from control_tree import CTNode, CTLeaf

N_ACTIONS = 2
N_ATTRIBS = 4

ATTRIBUTES = [("Cart Position", "continuous", -1, -1),
			  ("Cart Velocity", "continuous", -1, -1),
			  ("Pole Angle", "continuous", -1, -1),
			  ("Pole Angular Velocity", "continuous", -1, -1)]

def determine_split_mean(leaf):
	best_split = None
	best_mean = 0

	for attribute_idx in range(N_ATTRIBS):
		attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
		cutoffs = []

		if attr_type == "continuous":
			cutoffs = np.quantile([s[attribute_idx] for s in leaf.state_history], np.linspace(0.1, 0.9, 9))
		elif attr_type == "discrete":
			cutoffs = range(start_value, end_value)

		for cutoff in cutoffs:
			best_left_mean = 0
			for action in range(N_ACTIONS):
				L_partition = [dq for (s, dq) in leaf.history[action] if s[attribute_idx] <= cutoff]
				if len(L_partition) > 0:
					# left_mean = np.max([0, np.mean(L_partition)])
					left_mean = np.abs(np.mean(L_partition))
					# print(f"state i <= {cutoff}: action {'LEFT' if action == 0 else 'RIGHT'}, mean = |{'{:.4f}'.format(np.mean(L_partition))}|, std dev = {'{:.4f}'.format(np.std(L_partition))}")
					if left_mean > best_left_mean:
						best_left_mean = left_mean
				
			best_right_mean = 0
			for action in range(N_ACTIONS):
				R_partition = [dq for (s, dq) in leaf.history[action] if s[attribute_idx] > cutoff]
				if len(R_partition) > 0:
					# right_mean = np.max([0, np.mean(R_partition)])
					right_mean = np.abs(np.mean(R_partition))
					# print(f"state i > {cutoff}: action {'LEFT' if action == 0 else 'RIGHT'}, mean = |{'{:.4f}'.format(np.mean(R_partition))}|, std dev = {'{:.4f}'.format(np.std(R_partition))}")
					if right_mean > best_right_mean:
						best_right_mean = right_mean

			mean = best_left_mean + best_right_mean

			if mean > best_mean:
				# print(f"new best_split! i <= {cutoff}. mean: {'{:.4f}'.format(best_left_mean)} + {'{:.4f}'.format(best_right_mean)}")
				best_split = (attribute_idx, cutoff)
				best_mean = mean
	
	print(f"Split created: {best_split}")
	return best_split

def grow_tree(tree, leaf):
	split = determine_split_mean(leaf)

	new_node = PHCTNode(split, None, None)
	new_node.left = PHCTLeaf(parent=new_node, is_left=True)
	new_node.right = PHCTLeaf(parent=new_node, is_left=False)

	if leaf.parent is None:
		return new_node

	if leaf.is_left:
		leaf.parent.left = new_node
	else:
		leaf.parent.right = new_node
	
	return tree

class PHCTNode(CTNode):
	def reset_values(self):
		if self.left is not None:
			self.left.reset_values()
		if self.right is not None:
			self.right.reset_values()

class PHCTLeaf(CTLeaf):
	def __init__(self, parent=None, is_left=False):
		self.parent = parent
		self.is_left = is_left

		self.history = [[] for _ in range(N_ACTIONS)]
		self.q_history = [[] for _ in range(N_ACTIONS)]
		self.state_history = []
		self.q_values = np.zeros(N_ACTIONS)

	def print_tree(self, level=1):
		# print(" " * 2 * level, f"Q(s, left)  = {self.q_values[0]}")
		# print(" " * 2 * level, f"Q(s, right) = {self.q_values[1]}")
		print(" " * 2 * level, f"Q(s, left)  = {'{:.4f}'.format(self.q_values[0])}, mean ΔQ = {'---' if len(self.q_history[0]) == 0 else '{:.4f}'.format(np.mean([q for q in self.q_history[0]]))}, std dev ΔQ = {'---' if len(self.q_history[0]) == 0 else '{:.4f}'.format(np.std([q for q in self.q_history[0]]))}")
		print(" " * 2 * level, f"Q(s, right) = {'{:.4f}'.format(self.q_values[1])}, mean ΔQ = {'---' if len(self.q_history[1]) == 0 else '{:.4f}'.format(np.mean([q for q in self.q_history[1]]))}, std dev ΔQ = {'---' if len(self.q_history[1]) == 0 else '{:.4f}'.format(np.std([q for q in self.q_history[1]]))}")
	
	def predict(self, state):
		return self, np.argmax(self.q_values)
	
	def reset_values(self):
		self.history = [[] for _ in range(N_ACTIONS)]
		self.q_history = [[] for _ in range(N_ACTIONS)]
		self.q_values = np.zeros(N_ACTIONS)
		self.state_history = []

eps = 1
learning_rate = 0.05
discount_factor = 0.99

qtree = PHCTLeaf(parent=None)
has_grown = 0

total_rewards = []

env = gym.make("CartPole-v1")

for i_episode in range(1, 5000):
	state = env.reset()
	reward, total_reward = 0, 0
	done = False
	
	while not done:
		leaf, action = qtree.predict(state)

		# if np.random.random() < 0.1:
		if np.random.random() < max([0.1, (5000 / i_episode)]):
			action = np.random.randint(0, N_ACTIONS)

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = qtree.predict(next_state)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])
		
		leaf.q_values[action] += delta_q
		leaf.history[action].append((state, delta_q))
		leaf.state_history.append(state)
		leaf.q_history[action].append(delta_q)

		H = leaf.q_history[action]

		if len(H) > 5000 and len(H) % 100 == 0 and leaf.q_values[action] != 0:
			sigma = np.std(H)
			normalized_sigma = (sigma / leaf.q_values[action])

			if normalized_sigma > 0.005 * (has_grown):
			# if sigma > 0.2 * (has_grown * 1):
				print(f"In episode {i_episode} (len(H): {len(H)}), stdev for {('left' if leaf.is_left else 'right') if leaf.parent is not None else 'root'} leaf{(', child of ' + str((leaf.parent.attribute, leaf.parent.value))) if leaf.parent is not None else ''} (action {'left' if action == 0 else 'right'}) is {sigma}.")
				print(f"std dev of ΔQ / current Q-value = {sigma} / {leaf.q_values[action]} = {sigma / leaf.q_values[action]}")
				print("Average reward per episode:", np.mean(total_rewards))
				print("Current tree:")
				qtree.print_tree()
				qtree = grow_tree(qtree, leaf)
				qtree.reset_values()
				print("New tree:")
				qtree.print_tree()
				print("")

				total_rewards = []
				has_grown += 1
				i_episode = 1

		total_reward += reward
		state = next_state
	
	total_rewards.append(total_reward)

for i_episode in range(1, 5000):
	state = env.reset()
	reward, total_reward = 0, 0
	done = False
	
	while not done:
		leaf, action = qtree.predict(state)

		if np.random.random() < (500 / i_episode):
			action = np.random.randint(0, N_ACTIONS)

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = qtree.predict(next_state)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])
		
		leaf.q_values[action] += delta_q
		state = next_state

# Printing final result
print("=== FINAL TREE ===")
qtree.print_tree()
total_rewards = []
for i_episode in range(100):
	state = env.reset()
	reward, total_reward = 0, 0
	done = False
	
	while not done:
		leaf, action = qtree.predict(state)
		next_state, reward, done, _ = env.step(action)
		total_reward += reward
		state = next_state
	
	total_rewards.append(total_reward)
print("Average reward per episode:", np.mean(total_rewards))