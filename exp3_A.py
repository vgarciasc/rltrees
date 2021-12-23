import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from control_tree import CTNode, CTLeaf

def mdp_step(state, action, timestep):
	next_state, reward = state, 0
	done = False

	if action == 0 and state > 0:
		next_state -= 1
	elif action == 1 and state <= 3:
		next_state += 1
	
	# # MDP 1
	# if (state == 3 and next_state == 2) or (state == 2 and next_state == 3):
	# 	reward = 1
	# if state == 4 and action == 1:
	# 	done = True
	
	# # MDP 2
	# if (state == 0 and next_state == 1):
	# 	reward = 0.5
	# if (state == 3 and next_state == 4):
	# 	reward = 1
	# if state == 4 and action == 1:
	# 	done = True
	
	# MDP 3
	# if (state == 0 and next_state == 1):
	# 	reward = 1
	# if (state == 1 and next_state == 2):
	# 	reward = 1
	# if (state == 2 and next_state == 1):
	# 	reward = 0.1
	# if state == 4 and action == 1:
	# 	done = True
	
	# MDP 4
	if (state == 0 and next_state == 1):
		reward = 1
	if (state == 2 and next_state == 3):
		reward = 1
	if (state == 3 and next_state == 2):
		reward = 1
	if state == 4 and action == 1:
		done = True
	if state == 0 and action == 0:
		done = True

	if action == 1 and state == 4:
		done = True
	if timestep > 100:
		done = True
	
	return next_state, reward, done

class PHCTLeaf(CTLeaf):
	def __init__(self):
		self.history = []
		self.q_values = np.zeros(N_ACTIONS)

	def print_tree(self, level=1):
		print(" " * 2 * level, str(np.argmax(self.q_values)))
	
	def predict(self, state):
		return self, np.argmax(self.q_values)

env = gym.make("CartPole-v1")

N_ACTIONS = 2
N_STATE = 4

eps = 1
learning_rate = 0.1
discount_factor = 0.99

total_rewards = []
left_leaf_dataset = []
q_values = [0, 0]
history = []

for i_episode in range(5000):
	state = 0
	reward, total_reward = 0, 0
	done = False
	t = 0
	
	while not done:
		t += 1

		# action = np.random.choice([0, 1])
		# action = np.argmax(q_values) if np.random.random() > 0.1 else np.random.choice([0, 1])
		action = 1 if state <= 2 else 0

		next_state, reward, done = mdp_step(state, action, t)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(q_values) - q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - q_values[action])

		q_values[action] += delta_q
		history.append((state, action, delta_q))
		# history.append((state, action, q_values[action]))

		# print(f"state: {state}")
		# print(f"action: {action}")
		total_reward += reward
		state = next_state

	total_rewards.append(total_reward)
		
print("Average reward per episode:", np.mean(total_rewards))

print(f"Q(S_all, LEFT) = {q_values[0]}")
print(f"Q(S_all, RIGHT) = {q_values[1]}")

# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].hist([q for (i, a, q) in history if a == 0], color="blue", bins=3)
# axs[0].set_xlabel("$\Delta Q(s, LEFT)$")
# # axs[0].set_xlim(-1.5, 0.5)
# axs[0].set_title(f"$i \in {{0, 1, 2, 3, 4}}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if a == 0]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if a == 0]))}$")
# axs[1].hist([q for (i, a, q) in history if a == 1], color="orange", bins=3)
# axs[1].set_xlabel("$\Delta Q(s, RIGHT)$")
# # axs[1].set_xlim(-1.5, 0.5)
# axs[1].set_title(f"$i \in {{0, 1, 2, 3, 4}}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if a == 1]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if a == 1]))}$")
# plt.suptitle(f"$\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history]))}$")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 5, sharey=True, sharex=True)
# for attr_i in range(5):
# 	axs[0, attr_i].hist([q for (i, a, q) in history if a == 0 and i == attr_i], color="blue", bins=5)
# 	axs[0, attr_i].set_xlabel("$\Delta Q(s, LEFT)$")
# 	# axs[0, attr_i].set_xlim(-1.5, 0.5)
# 	axs[0, attr_i].set_title(f"i = {attr_i}")
# 	axs[1, attr_i].hist([q for (i, a, q) in history if a == 1 and i == attr_i], color="orange", bins=5)
# 	axs[1, attr_i].set_xlabel("$\Delta Q(s, RIGHT)$")
# 	# axs[1, attr_i].set_xlim(-1.5, 0.5)
# plt.show()

# fig, axs = plt.subplots(1, 2)
# axs[0].scatter(range(len([q for (i, a, q) in history if a == 0])), [q for (i, a, q) in history if a == 0], s=2, color="blue", label="LEFT")
# axs[0].set_ylabel("$\Delta Q(S_{all}, LEFT)$")
# axs[1].set_xlabel("Iterations")
# axs[1].scatter(range(len([q for (i, a, q) in history if a == 1])), [q for (i, a, q) in history if a == 1], s=2, color="orange", label="RIGHT")
# axs[1].set_ylabel("$\Delta Q(S_{all}, RIGHT)$")
# axs[1].set_xlabel("Iterations")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 4, sharey=True)
# for attr_i in range(0, 4):
# 	best_left_action = np.argmax([np.mean([q for (i, a, q) in history if i <= attr_i and a == 0]), np.mean([q for (i, a, q) in history if i <= attr_i and a == 1])])
# 	best_right_action = np.argmax([np.mean([q for (i, a, q) in history if i > attr_i and a == 0]), np.mean([q for (i, a, q) in history if i > attr_i and a == 1])])
# 	left_partition = [q for (i, a, q) in history if i <= attr_i and a == best_left_action]
# 	right_partition = [q for (i, a, q) in history if i > attr_i and a == best_right_action]

# 	axs[(attr_i // 2), (attr_i % 2)*2].hist(left_partition, color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2)*2].set_xlabel(f"$\Delta Q(s, {'LEFT' if best_left_action == 0 else 'RIGHT'})$")
# 	axs[(attr_i // 2), (attr_i % 2)*2].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2)*2].set_title(f"$i \leq {attr_i}, \mu = {'{:.4f}'.format(np.mean(left_partition))}$")
# 	axs[(attr_i // 2), (attr_i % 2)*2 + 1].hist(right_partition, color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2)*2 + 1].set_xlabel(f"$\Delta Q(s, {'LEFT' if best_right_action == 0 else 'RIGHT'})$")
# 	axs[(attr_i // 2), (attr_i % 2)*2 + 1].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2)*2 + 1].set_title(f"$i > {attr_i}, \mu = {'{:.4f}'.format(np.mean(right_partition))}$")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharex=True, sharey=True)
# for attr_i in range(0, 4):
# 	best_left_action = np.argmax([np.mean([q for (i, a, q) in history if i <= attr_i and a == 0]), np.mean([q for (i, a, q) in history if i <= attr_i and a == 1])])
# 	best_right_action = np.argmax([np.mean([q for (i, a, q) in history if i > attr_i and a == 0]), np.mean([q for (i, a, q) in history if i > attr_i and a == 1])])
# 	left_partition = [q for (i, a, q) in history if i <= attr_i and a == 0]
# 	right_partition = [q for (i, a, q) in history if i > attr_i and a == 1]

# 	axs[0, attr_i * 2].hist([q for (i, a, q) in history if i <= attr_i and a == 0], color="blue", bins=3)
# 	axs[0, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, LEFT)$")
# 	# axs[0, attr_i * 2].set_xlim(-0.5, 0.5)
# 	axs[0, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if i <= attr_i and a == 0]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if i <= attr_i and a == 0]))}$")
# 	axs[0, attr_i * 2 + 1].hist([q for (i, a, q) in history if i > attr_i and a == 0], color="blue", bins=3)
# 	axs[0, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, LEFT)$")
# 	# axs[0, attr_i * 2 + 1].set_xlim(-0.5, 0.5)
# 	axs[0, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if i > attr_i and a == 0]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if i > attr_i and a == 0]))}$")

# 	axs[1, attr_i * 2].hist([q for (i, a, q) in history if i <= attr_i and a == 1], color="orange", bins=3)
# 	axs[1, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, RIGHT)$")
# 	# axs[1, attr_i * 2].set_xlim(-0.5, 0.5)
# 	axs[1, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if i <= attr_i and a == 1]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if i <= attr_i and a == 1]))}$")
# 	axs[1, attr_i * 2 + 1].hist([q for (i, a, q) in history if i > attr_i and a == 1], color="orange", bins=3)
# 	axs[1, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, RIGHT)$")
# 	# axs[1, attr_i * 2 + 1].set_xlim(-0.5, 0.5)
# 	axs[1, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history if i > attr_i and a == 1]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history if i > attr_i and a == 1]))}$")
# plt.suptitle("Estimativas de $\Delta Q$ após splits")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharey=True)
# for attr_i in range(0, 4):
# 	best_left_action = np.argmax([np.mean([q for (i, a, q) in history if i <= attr_i and a == 0]), np.mean([q for (i, a, q) in history if i <= attr_i and a == 1])])
# 	best_right_action = np.argmax([np.mean([q for (i, a, q) in history if i > attr_i and a == 0]), np.mean([q for (i, a, q) in history if i > attr_i and a == 1])])
# 	left_partition = [q for (i, a, q) in history if i <= attr_i and a == best_left_action]
# 	right_partition = [q for (i, a, q) in history if i > attr_i and a == best_right_action]

# 	axs[(attr_i // 2), (attr_i % 2) * 4].hist([q for (i, a, q) in history if i <= attr_i and a == 0], color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_xlabel(f"$\Delta Q(s, LEFT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_xlim(-1.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_title(f"$i \leq {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].hist([q for (i, a, q) in history if i <= attr_i and a == 1], color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_xlabel(f"$\Delta Q(s, RIGHT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_xlim(-1.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_title(f"$i \leq {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].hist([q for (i, a, q) in history if i > attr_i and a == 0], color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_xlabel(f"$\Delta Q(s, LEFT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_xlim(-1.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_title(f"$i > {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].hist([q for (i, a, q) in history if i > attr_i and a == 1], color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_xlabel(f"$\Delta Q(s, RIGHT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_xlim(-1.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_title(f"$i > {attr_i}$")
# plt.show()