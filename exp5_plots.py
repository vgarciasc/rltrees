import gym
import pdb
import numpy as np
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

def run_algorithm1(qtree, MDP, n_episodes=1000, learning_enabled=True, growing_enabled=True):
	episode_rewards = []
	for T in range(1, n_episodes):
		state = 0
		reward = 0
		episode_reward = 0
		done = False
		t = 0
		
		while not done:
			t += 1
			leaf, action = eps_greedy(qtree, [state], (max([0.1, (5000 / T)])) if learning_enabled else 0)
			next_state, reward, done = MDP(state, action, t)
			next_leaf, _ = qtree.predict([next_state])
			
			if not done:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(next_leaf.q_values) - leaf.q_values[action])
			else:
				delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])
			
			if learning_enabled:
				leaf.record([state], action, delta_q)

			if growing_enabled and should_split_leaf(leaf, action, T):
				print("Current tree:"); qtree.print_tree()

				qtree = grow_tree(qtree, leaf, determine_split_mean)
				qtree.reset_all()

				print("New tree:"); qtree.print_tree(); print("")

			state = next_state
			episode_reward += reward

		episode_rewards.append(episode_reward)

	return qtree, episode_rewards

def should_split_leaf(leaf, action, episode):
	H = leaf.dq_history[action]

	if len(H) > 5000 and len(H) % 100 == 0 and leaf.q_values[action] != 0:
		sigma = np.std(H)
		normalized_sigma = (sigma / leaf.q_values[action])

		if normalized_sigma > 0.002:
		# if sigma > 0.2 * (has_grown * 1):
			print(f"In episode {episode} (len(H): {len(H)}), stdev for {('left' if leaf.is_left else 'right') if leaf.parent is not None else 'root'} leaf{(', child of ' + str((leaf.parent.attribute, leaf.parent.value))) if leaf.parent is not None else ''} (action {'left' if action == 0 else 'right'}) is {normalized_sigma}.")
			return True
	return False

def determine_split_mean(leaf, verbose=False):
	best_split = None
	best_mean = 0

	for attribute_idx in range(N_ATTRIBS):
		attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
		
		for cutoff in range(start_value, end_value):
			best_left_mean = 0
			for action in range(N_ACTIONS):
				L_partition = [dq for (s, dq) in leaf.full_dq_history[action] if s[attribute_idx] <= cutoff]
				if len(L_partition) > 0:
					# left_mean = np.max([0, np.mean(L_partition)])
					left_mean = np.abs(np.mean(L_partition))
					if verbose:
						print(f"state i <= {cutoff}: action {'LEFT' if action == 0 else 'RIGHT'}, mean = |{'{:.4f}'.format(np.mean(L_partition))}|, std dev = {'{:.4f}'.format(np.std(L_partition))}")
					if left_mean > best_left_mean:
						best_left_mean = left_mean
				
			best_right_mean = 0
			for action in range(N_ACTIONS):
				R_partition = [dq for (s, dq) in leaf.full_dq_history[action] if s[attribute_idx] > cutoff]
				if len(R_partition) > 0:
					# right_mean = np.max([0, np.mean(R_partition)])
					right_mean = np.abs(np.mean(R_partition))
					if verbose:
						print(f"state i > {cutoff}: action {'LEFT' if action == 0 else 'RIGHT'}, mean = |{'{:.4f}'.format(np.mean(R_partition))}|, std dev = {'{:.4f}'.format(np.std(R_partition))}")
					if right_mean > best_right_mean:
						best_right_mean = right_mean

			mean = best_left_mean + best_right_mean

			if mean > best_mean:
				if verbose:
					print(f"new best_split! i <= {cutoff}. mean: {'{:.4f}'.format(best_left_mean)} + {'{:.4f}'.format(best_right_mean)}")
				best_split = (attribute_idx, cutoff)
				best_mean = mean
	
	print(f"Split created: {best_split}")
	return best_split

# Setting up environment
MDP = mdp.mdp4_step

# Initializing tree
qtree = QLeaf(parent=None, actions=["left", "right"])

# Creating tree
qtree, _ = run_algorithm1(qtree, MDP, 15000, growing_enabled=False)

# Plot 1
# fig, axs = plt.subplots(2, 2, sharey='row')
# axs[0,0].plot(range(len(qtree.q_history[0])), qtree.q_history[0], color="blue")
# axs[0,0].set_ylabel("$Q(S_{all}, LEFT)$")
# axs[0,0].set_title("$i \in \{0, 1, 2, 3, 4\}$")
# axs[1,0].scatter(range(len(qtree.dq_history[0])), qtree.dq_history[0], s=2, color="blue")
# axs[1,0].set_ylabel("$\Delta Q(S_{all}, LEFT)$")
# axs[1,0].set_xlabel("Iterations")
# axs[0,1].plot(range(len(qtree.q_history[1])), qtree.q_history[1], color="orange")
# axs[0,1].set_ylabel("$Q(S_{all}, RIGHT)$")
# axs[0,1].set_title("$i \in \{0, 1, 2, 3, 4\}$")
# axs[1,1].scatter(range(len(qtree.dq_history[1])), qtree.dq_history[1], s=2, color="orange")
# axs[1,1].set_ylabel("$\Delta Q(S_{all}, RIGHT)$")
# axs[1,1].set_xlabel("Iterations")
# plt.show()

# Plot 2
# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharex=True, sharey=True)
# for attr_i in range(0, 4):
# 	partition = [dq for (s, dq) in qtree.full_dq_history[0] if s[0] <= attr_i]
# 	axs[0, attr_i * 2].hist(partition, color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, LEFT)$")
# 	axs[0, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")
	
# 	partition = [dq for (s, dq) in qtree.full_dq_history[0] if s[0] > attr_i]
# 	axs[0, attr_i * 2 + 1].hist(partition, color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, LEFT)$")
# 	axs[0, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")

# 	partition = [dq for (s, dq) in qtree.full_dq_history[1] if s[0] <= attr_i]
# 	axs[1, attr_i * 2].hist(partition, color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, RIGHT)$")
# 	axs[1, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")

# 	partition = [dq for (s, dq) in qtree.full_dq_history[1] if s[0] > attr_i]
# 	axs[1, attr_i * 2 + 1].hist(partition, color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, RIGHT)$")
# 	axs[1, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")
# plt.subplots_adjust(left=0.02, bottom=0.07, right=0.98, top=0.92, wspace=0.20, hspace=0.38)
# plt.show()

# Plot 3
# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharex=True, sharey=True)
# for attr_i in range(0, 4):
# 	partition = [dq for (s, dq) in qtree.full_q_history[0] if s[0] <= attr_i]
# 	axs[0, attr_i * 2].hist(partition, color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2].set_xlabel(f"$Q(s_{{i \leq {attr_i}}}, LEFT)$")
# 	axs[0, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")
	
# 	partition = [dq for (s, dq) in qtree.full_q_history[0] if s[0] > attr_i]
# 	axs[0, attr_i * 2 + 1].hist(partition, color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2 + 1].set_xlabel(f"$Q(s_{{i > {attr_i}}}, LEFT)$")
# 	axs[0, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")

# 	partition = [dq for (s, dq) in qtree.full_q_history[1] if s[0] <= attr_i]
# 	axs[1, attr_i * 2].hist(partition, color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2].set_xlabel(f"$Q(s_{{i \leq {attr_i}}}, RIGHT)$")
# 	axs[1, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")

# 	partition = [dq for (s, dq) in qtree.full_q_history[1] if s[0] > attr_i]
# 	axs[1, attr_i * 2 + 1].hist(partition, color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2 + 1].set_xlabel(f"$Q(s_{{i > {attr_i}}}, RIGHT)$")
# 	axs[1, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean(partition))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(partition))}$")
# plt.subplots_adjust(left=0.02, bottom=0.07, right=0.98, top=0.92, wspace=0.20, hspace=0.38)
# plt.show()

# Initializing tree
qtree = QLeaf(parent=None, actions=["left", "right"])

# Creating tree
qtree, _ = run_algorithm1(qtree, MDP, 5000, growing_enabled=True)

# Converging to Q*
qtree, _ = run_algorithm1(qtree, MDP, 30000, growing_enabled=False)

# Printing final result
print("=== FINAL TREE ==="); qtree.print_tree()
qtree, episode_rewards = run_algorithm1(qtree, MDP, 1000, growing_enabled=False, learning_enabled=False)
print("Average reward per episode:", np.mean(episode_rewards))

# Plot 4
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(2, 4, sharey='row')

test = stats.kstest((qtree.left.left.q_history[1] - np.mean(qtree.left.left.q_history[1][-1000:]) / np.std(qtree.left.left.q_history[1][-1000:])), 'norm')
axs[0,0].set_title(f"$i \in \{{0\}}$ \n $\mu = {'{:.4f}'.format(np.mean(qtree.left.left.q_history[1][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.left.q_history[1][-1000:]))}$")
axs[0,0].plot(range(len(qtree.left.left.q_history[1])), qtree.left.left.q_history[1], color="orange")
axs[0,0].set_ylabel("$Q(S_{all}, RIGHT)$")
axs[1,0].set_title(f"$\mu = {'{:.4f}'.format(np.mean(qtree.left.left.dq_history[1]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.left.dq_history[1]))}$ \n $D = {'{:.4f}'.format(test[0])}, p = {'{:.4f}'.format(test[1])}$")
axs[1,0].scatter(range(len(qtree.left.left.dq_history[1])), qtree.left.left.dq_history[1], s=2, color="orange")
axs[1,0].set_ylabel("$\Delta Q(S_{all}, RIGHT)$")

test = stats.kstest((qtree.left.right.q_history[0] - np.mean(qtree.left.right.q_history[0][-1000:]) / np.std(qtree.left.right.q_history[0][-1000:])), 'norm')
axs[0,1].set_title(f"$i \in \{{1, 2, 3\}}$ \n $\mu = {'{:.4f}'.format(np.mean(qtree.left.right.q_history[0][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.right.q_history[0][-1000:]))}$")
axs[0,1].plot(range(len(qtree.left.right.q_history[0])), qtree.left.right.q_history[0], color="blue")
axs[0,1].set_ylabel("$Q(S_{all}, LEFT)$")
axs[1,1].set_title(f"$\mu = {'{:.4f}'.format(np.mean(qtree.left.right.dq_history[0][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.right.dq_history[0][-1000:]))}$ \n $D = {'{:.4f}'.format(test[0])}, p = {'{:.4f}'.format(test[1])}$")
axs[1,1].scatter(range(len(qtree.left.right.dq_history[0])), qtree.left.right.dq_history[0], s=2, color="blue")
axs[1,1].set_ylabel("$\Delta Q(S_{all}, LEFT)$")

test = stats.kstest((qtree.left.right.q_history[1] - np.mean(qtree.left.right.q_history[1][-1000:]) / np.std(qtree.left.right.q_history[1][-1000:])), 'norm')
axs[0,2].set_title(f"$i \in \{{1, 2, 3\}}$ \n $\mu = {'{:.4f}'.format(np.mean(qtree.left.right.q_history[1][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.right.q_history[1][-1000:]))}$")
axs[0,2].plot(range(len(qtree.left.right.q_history[1])), qtree.left.right.q_history[1], color="orange")
axs[0,2].set_ylabel("$Q(S_{all}, RIGHT)$")
axs[1,2].set_title(f"$\mu = {'{:.4f}'.format(np.mean(qtree.left.right.dq_history[1][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.left.right.dq_history[1][-1000:]))}$ \n $D = {'{:.4f}'.format(test[0])}, p = {'{:.4f}'.format(test[1])}$")
axs[1,2].scatter(range(len(qtree.left.right.dq_history[1])), qtree.left.right.dq_history[1], s=2, color="orange")
axs[1,2].set_ylabel("$\Delta Q(S_{all}, RIGHT)$")

test = stats.kstest((qtree.right.q_history[0] - np.mean(qtree.right.q_history[0][-1000:]) / np.std(qtree.right.q_history[0][-1000:])), 'norm')
axs[0,3].set_title(f"$i \in \{{4\}}$ \n $\mu = {'{:.4f}'.format(np.mean(qtree.right.q_history[0][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.right.q_history[0][-1000:]))}$")
axs[0,3].plot(range(len(qtree.right.q_history[0])), qtree.right.q_history[0], color="blue")
axs[0,3].set_ylabel("$Q(S_{all}, LEFT)$")
axs[1,3].set_title(f"$\mu = {'{:.4f}'.format(np.mean(qtree.right.dq_history[0][-1000:]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var(qtree.right.dq_history[0][-1000:]))}$ \n $D = {'{:.4f}'.format(test[0])}, p = {'{:.4f}'.format(test[1])}$")
axs[1,3].scatter(range(len(qtree.right.dq_history[0])), qtree.right.dq_history[0], s=2, color="blue")
axs[1,3].set_ylabel("$\Delta Q(S_{all}, LEFT)$")
plt.show()