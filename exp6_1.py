import gym
import copy
import pdb
import numpy as np
from numpy.lib.function_base import average
from scipy.stats.stats import ks_2samp
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
			leaf, action = eps_greedy(qtree, [state], 1)
			next_state, reward, done = MDP(state, action, t)
			next_leaf, _ = qtree.predict([next_state])

			leaf.q_history[0].append(([state], action, next_leaf.value, reward))

			state = next_state
	
	return qtree

def update_datapoints(node):
	if node.__class__.__name__ == "QLeaf":
		leaf = node
		
		for (s, a, v, r) in leaf.q_history[0]:
			q = r + DISCOUNT_FACTOR * v
			leaf.full_q_history[0].append((s, a, q))
	else:
		if node.left is not None:
			update_datapoints(node.left)
		if node.right is not None:
			update_datapoints(node.right)

	return node

def make_splits(qtree, node):
	if node.__class__.__name__ == "QLeaf":
		leaf = node
		print(f"\n{'Left' if leaf.is_left else 'Right'} leaf{(' of x[' + str(leaf.parent.attribute) + '] <= ' + str(leaf.parent.value)) if leaf.parent is not None else ''}:")

		best_split = None
		best_score = (0, 1)

		for attribute_idx in range(N_ATTRIBS):
			attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
			
			for cutoff in range(start_value, end_value):
				L_partition = [q for (s, a, q) in leaf.full_q_history[0] if s[attribute_idx] <= cutoff]
				R_partition = [q for (s, a, q) in leaf.full_q_history[0] if s[attribute_idx] > cutoff]

				if len(L_partition) > 0 and len(R_partition) > 0:
					kstest = stats.ks_2samp(L_partition, R_partition)
					score = kstest
					print(f"> Split {(attr_name, cutoff)} has score {kstest}")

					if (score[1] < best_score[1]) or (score[1] == best_score[1] and score[0] > best_score[0]):
					# if score < best_score:
						best_split = (attribute_idx, cutoff)
						best_score = score
		
		print(f">> Best split is {best_split}, with score {best_score}")
		if best_score[1] < 0.05:
			print(f">> Split {best_split} is good enough!")
			qtree = grow_tree(qtree, leaf, None, best_split)
	else:
		if node.left is not None:
			make_splits(qtree, node.left)
		if node.right is not None:
			make_splits(qtree, node.right)

	return qtree

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
				L_partition = [q for (s, a, q) in leaf.full_q_history[0] if s[attribute_idx] <= cutoff]
				R_partition = [q for (s, a, q) in leaf.full_q_history[0] if s[attribute_idx] > cutoff]

				if len(L_partition) > 0 and len(R_partition) > 0:
					kstest = stats.ks_2samp(L_partition, R_partition)
					score = kstest
					print(f"> Split {(attr_name, cutoff)} has score {kstest}")

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

def run_monte_carlo_control(qtree, n_episodes=1000):
	for T in range(1, n_episodes):
		episode = []
		state = 0
		t = 0
		done = False

		while not done:
			t += 1
			leaf, action = qtree.predict([state])

			if np.random.random() < 0.5 * n_episodes / T:
				action = np.random.randint(0, N_ACTIONS)
			
			next_state, reward, done = MDP(state, action, t)
			episode.append((state, action, reward))
			state = next_state

		G = 0
		for t in range(len(episode) - 1, -1, -1):
			state, action, reward = episode[t]
			leaf, _ = qtree.predict([state])
			
			G = DISCOUNT_FACTOR * G + reward

			has_appeared = False
			for i in range(0, t):
				state2, action2, _ = episode[i]
				if state == state2 and action == action2:
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
MDP = mdp.mdp1_step

# Initializing tree
qtree = QLeaf(parent=None, actions=["left", "right"])
best_reward = 0

# Data collecting phase
qtree = collect_data(qtree, 10000)

# Split phase
qtree = update_datapoints(qtree)

fig, axs = plt.subplots(2, 5, sharex='row')
for attr_id in [0, 1, 2, 3, 4]:
	partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] == attr_id]
	axs[0, attr_id].hist(partition, bins=3)
	axs[0, attr_id].set_title(f"$i = {attr_id}$")
	axs[0, attr_id].set_ylabel("Count")
	axs[0, attr_id].set_xlabel("$q(I, a)$")

	values, base = np.histogram(partition, bins=40)
	axs[1, attr_id].plot(base[:-1], np.cumsum(values) / np.cumsum(values)[-1])
	axs[1, attr_id].set_ylim(bottom=0)
	axs[1, attr_id].set_ylabel("CDF")
	axs[1, attr_id].set_xlabel("$q(I, a)$")
plt.show()

fig, axs = plt.subplots(2, 4, sharex='row')
for attr_id in [0, 1, 2, 3]:
	l_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] <= attr_id]
	r_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] > attr_id]
	axs[0, attr_id].hist(l_partition, bins=5, alpha=0.7, color="green", label=f"$i \leq {attr_id}$")
	axs[0, attr_id].hist(r_partition, bins=5, alpha=0.7, color="red", label=f"$i > {attr_id}$")
	axs[0, attr_id].legend()
	axs[0, attr_id].set_title(f"$attr = {attr_id}$")
	axs[0, attr_id].set_xlabel("$q(I, a)$")

	l_values, l_base = np.histogram(l_partition, bins=40)
	r_values, r_base = np.histogram(r_partition, bins=40)
	axs[1, attr_id].plot(l_base[:-1], np.cumsum(l_values) / np.cumsum(l_values)[-1], linestyle='dashed', color="green", label=f"$i \leq {attr_id}$")
	axs[1, attr_id].plot(r_base[:-1], np.cumsum(r_values) / np.cumsum(r_values)[-1], linestyle='dashed', color="red", label=f"$i > {attr_id}$")
	axs[1, attr_id].legend()
	axs[1, attr_id].set_title(f"$D = {'{:.4f}'.format(ks_2samp(l_partition, r_partition)[0])}, p = {'{:.4f}'.format(ks_2samp(l_partition, r_partition)[1])}$")
	axs[1, attr_id].set_ylim(bottom=0)
	axs[1, attr_id].set_ylabel("CDF")
	axs[1, attr_id].set_xlabel("$q(I, a)$")
plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharex='row')
# for attr_id in [0, 1, 2, 3]:
# 	ll_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] <= attr_id and a == 0]
# 	lr_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] <= attr_id and a == 1]
# 	rl_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] > attr_id and a == 0]
# 	rr_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] > attr_id and a == 1]

# 	axs[0, (attr_id * 2)].hist(ll_partition, bins=5, alpha=0.7, color="blue", label=f"$i \leq {attr_id}$")
# 	axs[0, (attr_id * 2)].hist(rl_partition, bins=5, alpha=0.7, color="cyan", label=f"$i > {attr_id}$")
# 	axs[0, (attr_id * 2)].legend()
# 	axs[0, (attr_id * 2)].set_title(f"$LEFT$")
# 	axs[0, (attr_id * 2)].set_xlabel("$q(I, a)$")

# 	axs[0, (attr_id * 2) + 1].hist(lr_partition, bins=5, alpha=0.7, color="orange", label=f"$i \leq {attr_id}$")
# 	axs[0, (attr_id * 2) + 1].hist(rr_partition, bins=5, alpha=0.7, color="red", label=f"$i > {attr_id}$")
# 	axs[0, (attr_id * 2) + 1].legend()
# 	axs[0, (attr_id * 2) + 1].set_title(f"$RIGHT$")
# 	axs[0, (attr_id * 2) + 1].set_xlabel("$q(I, a)$")

# 	ll_values, ll_base = np.histogram(ll_partition, bins=40)
# 	lr_values, lr_base = np.histogram(lr_partition, bins=40)
# 	rl_values, rl_base = np.histogram(rl_partition, bins=40)
# 	rr_values, rr_base = np.histogram(rr_partition, bins=40)

# 	axs[1, (attr_id * 2)].plot(ll_base[:-1], np.cumsum(ll_values) / np.cumsum(ll_values)[-1], linestyle='dashed', color="blue", label=f"$i \leq {attr_id}$")
# 	axs[1, (attr_id * 2)].plot(rl_base[:-1], np.cumsum(rl_values) / np.cumsum(rl_values)[-1], linestyle='dashed', color="cyan", label=f"$i > {attr_id}$")
# 	axs[1, (attr_id * 2)].legend()
# 	axs[1, (attr_id * 2)].set_title(f"$D = {'{:.4f}'.format(ks_2samp(ll_partition, rl_partition)[0])}, p = {'{:.4f}'.format(ks_2samp(ll_partition, rl_partition)[1])}$")
# 	axs[1, (attr_id * 2)].set_ylim(bottom=0)
# 	axs[1, (attr_id * 2)].set_xlabel("$q(I, a)$")

# 	axs[1, (attr_id * 2) + 1].plot(lr_base[:-1], np.cumsum(lr_values) / np.cumsum(lr_values)[-1], linestyle='dashed', color="orange", label=f"$i \leq {attr_id}$")
# 	axs[1, (attr_id * 2) + 1].plot(rr_base[:-1], np.cumsum(rr_values) / np.cumsum(rr_values)[-1], linestyle='dashed', color="red", label=f"$i > {attr_id}$")
# 	axs[1, (attr_id * 2) + 1].legend()
# 	axs[1, (attr_id * 2) + 1].set_title(f"$D = {'{:.4f}'.format(ks_2samp(lr_partition, rr_partition)[0])}, p = {'{:.4f}'.format(ks_2samp(lr_partition, rr_partition)[1])}$")
# 	axs[1, (attr_id * 2) + 1].set_ylim(bottom=0)
# 	axs[1, (attr_id * 2) + 1].set_xlabel("$q(I, a)$")
# plt.show()