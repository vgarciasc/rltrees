import copy
import mdp
import pdb
import gym
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.99

N_ACTIONS = 2
N_ATTRIBS = 4
ATTRIBUTES = [("Cart Position", "continuous", -1, -1),
			  ("Cart Velocity", "continuous", -1, -1),
			  ("Pole Angle", "continuous", -1, -1),
			  ("Pole Angular Velocity", "continuous", -1, -1)]

history = []

def collect_data(qtree, n_episodes):
	for T in range(1, n_episodes):
		state = 0
		action = 0
		reward = 0
		leaf = None
		done = False
		t = 0
		
		while not done:
			t += 1

			if leaf is None:
				leaf, action = qtree.predict([state])

			if np.random.random() < max([0.2, ((0.1 / 2) * n_episodes / T)]):
				action = np.random.randint(0, N_ACTIONS)
			
			next_state, reward, done = MDP(state, action, t)
			next_leaf, next_action = qtree.predict([next_state])

			if done:
				reward = 0

			leaf.q_history[action].append((state, action, next_leaf.value, reward))
			leaf.state_history.append(state)

			leaf = next_leaf
			state = next_state
			action = next_action
	
	return qtree

def merge_leaves_same_action(qtree, node):
	if node.left.__class__.__name__ == "QNode":
		qtree = merge_leaves_same_action(qtree, node.left)
	if node.right.__class__.__name__ == "QNode":
		qtree = merge_leaves_same_action(qtree, node.right)
	
	if node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QLeaf":
		left_action = np.argmax(node.left.q_values)
		right_action = np.argmax(node.right.q_values)

		if left_action == right_action:
			merged_q_values = [node.left.q_values, node.right.q_values][np.argmax([np.max(node.left.q_values), np.max(node.right.q_values)])]
			merged_leaf = QLeaf(node.parent, node.parent is not None and node == node.parent.left, node.left.actions, merged_q_values)
			print(f"Merged leaves of node '{ATTRIBUTES[node.attribute][0]} <= {node.value}'!")
			if node.parent is not None:
				if node == node.parent.left:
					node.parent.left = merged_leaf
				else:
					node.parent.right = merged_leaf
	
	return qtree

def merge_leaves_reward(qtree, node, n_trials=100):
	if node.left.__class__.__name__ == "QNode":
		qtree = merge_leaves_reward(qtree, node.left)
	if node.right.__class__.__name__ == "QNode":
		qtree = merge_leaves_reward(qtree, node.right)
	
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
		print(f"Merged leaves of node '{ATTRIBUTES[node.attribute][0]} <= {node.value}'!")
		if node.parent is not None:
			print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
		
		new_average_reward = get_average_reward(qtree, n_trials)
		print(f"Got average reward {new_average_reward} after merge.")

		if new_average_reward < 0.9 * average_reward:
			print("Average reward was reduced too much. Undoing merge...")
			if node.parent is not None:
				if was_left == True:
					node.parent.left = node
				else:
					node.parent.right = node
		
			new_average_reward = get_average_reward(qtree, n_trials)
			print(f"Undid merge, got average reward {new_average_reward}.")
			if node.parent is not None:
				print(f"Now '{ATTRIBUTES[node.attribute][0]} <= {node.value}' has parent '{ATTRIBUTES[node.parent.attribute][0]} <= {node.parent.value}'")
		else:
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

		if new_average_reward < 0.9 * average_reward:
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
			history.append((qtree.get_size(), new_average_reward))

	return qtree

def get_average_reward(qtree, n_episodes):
	env = gym.make("CartPole-v1")
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

qtree = None
with open('data/saved_tree1', 'rb') as file:
# with open('data/cartpole_tree_pruned', 'rb') as file:
	qtree = pickle.load(file)
	file.close()

print("Original tree:")
qtree.print_tree()

average_reward = get_average_reward(qtree, 1000)
print(f"Average reward per episode = {average_reward}")

for _ in range(3):
	qtree = merge_leaves_reward(qtree, qtree, 5000)

average_reward = get_average_reward(qtree, 1000)
print(f"Average reward per episode = {average_reward}")

print("Final tree:")
qtree.print_tree()

with open('data/cartpole_tree_pruned', 'wb') as file:
	pickle.dump(qtree, file)
	print("> Saved best tree to file 'data/cartpole_tree_pruned'!")

plt.gca().invert_xaxis()
plt.plot([a for (a, b) in history], [b for (a, b) in history], color="blue")
plt.xlabel("Tree size")
plt.ylabel("Reward")
plt.title("Average reward during pruning")
plt.show()