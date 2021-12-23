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

	if (state == 2 and action == 1) or (state == 3 and action == 0):
		reward = 1

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

history = []
best_actions = []

for attr_i in range(0, 4):
	total_rewards = []
	q_values = [[0, 0], [0, 0]]

	history.append([[], []])
	history_q_values = [[[], []], [[], []]]

	for i_episode in range(5000):
		state = 0
		reward, total_reward = 0, 0
		done = False
		t = 0
		
		while not done:
			t += 1

			leaf = (0 if state <= attr_i else 1)

			# action = np.random.choice([0, 1])
			# action = np.argmax(q_values) if np.random.random() < 0.1 else np.random.choice([0, 1])
			action = np.argmax(q_values[leaf]) if np.random.random() > (1 / (i_episode + 1)) else np.random.choice([0, 1])

			next_state, reward, done = mdp_step(state, action, t)
			
			if not done:
				delta_q = learning_rate * (reward + discount_factor * np.max(q_values[0 if next_state <= attr_i else 1]) - q_values[leaf][action])
			else:
				delta_q = learning_rate * (reward + discount_factor * 0 - q_values[leaf][action])

			q_values[leaf][action] += delta_q
			# history[attr_i][leaf].append((state, action, q_values[leaf][action]))
			history[attr_i][leaf].append((state, action, delta_q))
			history_q_values[leaf][action].append(q_values[leaf][action])

			# print(f"state: {state}")
			# print(f"action: {action}")
			total_reward += reward
			state = next_state

		total_rewards.append(total_reward)
		
	print(f"i <= {attr_i}")
	print("Average reward per episode:", np.mean(total_rewards))

	print(f"Q(S_(i <= {attr_i}), a) = {['{:.4f}'.format(q) for q in q_values[0]]}, mean ΔQ = [{'{:.4f}'.format(np.mean([q for (_, a, q) in history[attr_i][0] if a == 0]))}, {'{:.4f}'.format(np.mean([q for (_, a, q) in history[attr_i][0] if a == 1]))}], std dev ΔQ = [{'{:.4f}'.format(np.std([q for (_, a, q) in history[attr_i][0] if a == 0]))}, {'{:.4f}'.format(np.std([q for (_, a, q) in history[attr_i][0] if a == 1]))}]")
	print(f"Q(S_(i >  {attr_i}), a) = {['{:.4f}'.format(q) for q in q_values[1]]}, mean ΔQ = [{'{:.4f}'.format(np.mean([q for (_, a, q) in history[attr_i][1] if a == 0]))}, {'{:.4f}'.format(np.mean([q for (_, a, q) in history[attr_i][1] if a == 1]))}], std dev ΔQ = [{'{:.4f}'.format(np.std([q for (_, a, q) in history[attr_i][1] if a == 0]))}, {'{:.4f}'.format(np.std([q for (_, a, q) in history[attr_i][1] if a == 1]))}]")
	print("")

	# best_action_left = np.argmax([np.mean(history_q_values[0][0][-100:]), np.mean(history_q_values[0][1][-100:])])
	# best_action_right = np.argmax([np.mean(history_q_values[1][0][-100:]), np.mean(history_q_values[1][1][-100:])])
	best_action_left = np.argmax([q_values[0]])
	best_action_right = np.argmax([q_values[1]])
	best_actions.append((best_action_left, best_action_right))

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 4)
# for attr_i in range(0, 4):
# 	best_left_action = best_actions[attr_i][0]
# 	best_right_action = best_actions[attr_i][1]
# 	left_partition = [q for (i, a, q) in history[attr_i][0] if a == best_left_action]
# 	right_partition = [q for (i, a, q) in history[attr_i][1] if a == best_right_action]

# 	print(f"if (i <= {attr_i}) then {'LEFT' if best_left_action == 0 else 'RIGHT'} else {'LEFT' if best_right_action == 0 else 'RIGHT'}")
# 	print(f"\t mean of ΔQ(s_(i <= {attr_i}), {'LEFT' if best_left_action == 0 else 'RIGHT'}): {np.mean([q for (_, a, q) in history[attr_i][0] if a == best_left_action])}")
# 	print(f"\t mean of ΔQ(s_(i <= {attr_i}), {'LEFT' if best_left_action == 1 else 'RIGHT'}): {np.mean([q for (_, a, q) in history[attr_i][0] if a == (1 if best_left_action == 0 else 0)])}")
# 	print(f"\t mean of ΔQ(s_(i > {attr_i}), {'LEFT' if best_right_action == 0 else 'RIGHT'}): {np.mean([q for (_, a, q) in history[attr_i][1] if a == best_right_action])}")
# 	print(f"\t mean of ΔQ(s_(i > {attr_i}), {'LEFT' if best_right_action == 1 else 'RIGHT'}): {np.mean([q for (_, a, q) in history[attr_i][1] if a == (1 if best_right_action == 0 else 0)])}")

# 	axs[(attr_i // 2), (attr_i % 2) * 2].hist(left_partition, color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 2].set_xlabel(f"$\Delta Q(s, {'LEFT' if best_left_action == 0 else 'RIGHT'})$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 2].set_title(f"$i \leq {attr_i}, \mu = {'{:.5f}'.format(np.mean([q for (_, a, q) in history[attr_i][0] if a == best_left_action]))}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].hist(right_partition, color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].set_xlabel(f"$\Delta Q(s, {'LEFT' if best_right_action == 0 else 'RIGHT'})$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].set_title(f"$i > {attr_i}, \mu = {'{:.5f}'.format(np.mean([q for (_, a, q) in history[attr_i][1] if a == best_right_action]))}$")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharey=True)
# for attr_i in range(0, 4):
# 	axs[0, attr_i * 2].hist([q for (i, a, q) in history[attr_i][0] if a == 0], color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, LEFT)$")
# 	# axs[0, attr_i * 2].set_xlim(-0.5, 0.5)
# 	axs[0, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][0] if a == 0]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history[attr_i][0] if a == 0]))}$")
	
# 	axs[0, attr_i * 2 + 1].hist([q for (i, a, q) in history[attr_i][1] if a == 0], color="blue", bins=3, density=True)
# 	axs[0, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, LEFT)$")
# 	# axs[0, attr_i * 2 + 1].set_xlim(-0.5, 0.5)
# 	axs[0, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][1] if a == 0]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history[attr_i][1] if a == 0]))}$")

# 	axs[1, attr_i * 2].hist([q for (i, a, q) in history[attr_i][0] if a == 1], color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, RIGHT)$")
# 	# axs[1, attr_i * 2].set_xlim(-0.5, 0.5)
# 	axs[1, attr_i * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][0] if a == 1]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history[attr_i][0] if a == 1]))}$")

# 	axs[1, attr_i * 2 + 1].hist([q for (i, a, q) in history[attr_i][1] if a == 1], color="orange", bins=3, density=True)
# 	axs[1, attr_i * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, RIGHT)$")
# 	# axs[1, attr_i * 2 + 1].set_xlim(-0.5, 0.5)
# 	axs[1, attr_i * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][1] if a == 1]))}$ \n $\sigma = {'{:.4f}'.format(np.std([q for (i, a, q) in history[attr_i][1] if a == 1]))}$")

# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 8, sharex=True, sharey=True)
# for attr_i in range(0, 4):
# 	axs[(attr_i // 2), (attr_i % 2) * 4].hist([q for (i, a, q) in history[attr_i][0] if a == 0], color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_xlabel(f"$\Delta Q(s, LEFT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4].set_title(f"$i \leq {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].hist([q for (i, a, q) in history[attr_i][0] if a == 1], color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_xlabel(f"$\Delta Q(s, RIGHT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 1].set_title(f"$i \leq {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].hist([q for (i, a, q) in history[attr_i][1] if a == 0], color="green", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_xlabel(f"$\Delta Q(s, LEFT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 2].set_title(f"$i > {attr_i}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].hist([q for (i, a, q) in history[attr_i][1] if a == 1], color="red", bins=3, density=True)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_xlabel(f"$\Delta Q(s, RIGHT)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_xlim(-0.5, 0.5)
# 	axs[(attr_i // 2), (attr_i % 2) * 4 + 3].set_title(f"$i > {attr_i}$")
# plt.show()

# plt.rcParams.update({'font.size': 8})
# fig, axs = plt.subplots(2, 4)
# for attr_i in range(0, 4):
# 	best_left_action = best_actions[attr_i][0]
# 	best_right_action = best_actions[attr_i][1]
# 	left_partition = [q for (i, a, q) in history[attr_i][0] if a == best_left_action]
# 	right_partition = [q for (i, a, q) in history[attr_i][1] if a == best_right_action]

# 	print(f"if (i <= {attr_i}) then {'LEFT' if best_left_action == 0 else 'RIGHT'} else {'LEFT' if best_right_action == 0 else 'RIGHT'}")
# 	print(f"\t mean of ΔQ(s_(i <= {attr_i}), {'LEFT' if best_left_action == 0 else 'RIGHT'}): {np.mean(history[attr_i][0][best_left_action])}")
# 	print(f"\t mean of ΔQ(s_(i <= {attr_i}), {'LEFT' if best_left_action == 1 else 'RIGHT'}): {np.mean(history[attr_i][0][0 if best_left_action == 1 else 1])}")
# 	print(f"\t mean of ΔQ(s_(i > {attr_i}), {'LEFT' if best_right_action == 0 else 'RIGHT'}): {np.mean(history[attr_i][1][best_right_action])}")
# 	print(f"\t mean of ΔQ(s_(i > {attr_i}), {'LEFT' if best_right_action == 1 else 'RIGHT'}): {np.mean(history[attr_i][1][0 if best_right_action == 1 else 1])}")

# 	axs[(attr_i // 2), (attr_i % 2) * 2].scatter(range(len([q for (i, a, q) in history[attr_i][0] if a == 0])), [q for (i, a, q) in history[attr_i][0] if a == 0], color="blue", label="LEFT", s=2)
# 	axs[(attr_i // 2), (attr_i % 2) * 2].scatter(range(len([q for (i, a, q) in history[attr_i][0] if a == 1])), [q for (i, a, q) in history[attr_i][0] if a == 1], color="orange", label="RIGHT", s=2)
# 	axs[(attr_i // 2), (attr_i % 2) * 2].set_xlabel(f"$\Delta Q(s, a)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2].set_title(f"$i \leq {attr_i}, best = {'LEFT' if best_left_action == 0 else 'RIGHT'}, \mu = {'{:.2f}'.format(np.mean(history[attr_i][0][best_left_action]))}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2].legend()
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].scatter(range(len([q for (i, a, q) in history[attr_i][1] if a == 0])), [q for (i, a, q) in history[attr_i][1] if a == 0], color="blue", label="LEFT", s=2)
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].scatter(range(len([q for (i, a, q) in history[attr_i][1] if a == 1])), [q for (i, a, q) in history[attr_i][1] if a == 1], color="orange", label="RIGHT", s=2)
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].set_xlabel(f"$\Delta Q(s, a)$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].set_title(f"$i > {attr_i}, best = {'LEFT' if best_right_action == 0 else 'RIGHT'}, \mu = {'{:.2f}'.format(np.mean(history[attr_i][1][best_right_action]))}$")
# 	axs[(attr_i // 2), (attr_i % 2) * 2 + 1].legend()
# plt.show()

# fig, axs = plt.subplots(2, 2, sharey=True)
# axs[0,0].scatter(range(len([q for (i, a, q) in history[2][0] if a == 0])), [q for (i, a, q) in history[2][0] if a == 0], s=2, color="blue")
# axs[0,0].set_ylabel("$\Delta Q(S_{i \leq 2}, LEFT)$")
# axs[0,0].set_title("$i \leq 2$")
# axs[0,1].scatter(range(len([q for (i, a, q) in history[2][1] if a == 0])), [q for (i, a, q) in history[2][1] if a == 0], s=2, color="blue")
# axs[0,1].set_ylabel("$\Delta Q(S_{i > 2}, LEFT)$")
# axs[0,1].set_title("$i > 2$")
# axs[1,0].scatter(range(len([q for (i, a, q) in history[2][0] if a == 1])), [q for (i, a, q) in history[2][0] if a == 1], s=2, color="orange")
# axs[1,0].set_ylabel("$\Delta Q(S_{i \leq 2}, RIGHT)$")
# axs[1,0].set_xlabel("Iterations")
# axs[1,1].scatter(range(len([q for (i, a, q) in history[2][1] if a == 1])), [q for (i, a, q) in history[2][1] if a == 1], s=2, color="orange")
# axs[1,1].set_ylabel("$\Delta Q(S_{i > 2}, RIGHT)$")
# axs[1,1].set_xlabel("Iterations")
# plt.show()

plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(2, 4, sharey=True)
for attr_i in [2, 3]:
	axs[0, (attr_i - 2) * 2].hist([q for (i, a, q) in history[attr_i][0] if a == 0], color="blue", bins=3, density=True)
	axs[0, (attr_i - 2) * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, LEFT)$")
	# axs[0, (attr_i - 2) * 2].set_xlim(-0.5, 0.5)
	axs[0, (attr_i - 2) * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][0] if a == 0]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var([q for (i, a, q) in history[attr_i][0] if a == 0]))}$")
	
	axs[0, (attr_i - 2) * 2 + 1].hist([q for (i, a, q) in history[attr_i][1] if a == 0], color="blue", bins=3, density=True)
	axs[0, (attr_i - 2) * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, LEFT)$")
	# axs[0, (attr_i - 2) * 2 + 1].set_xlim(-0.5, 0.5)
	axs[0, (attr_i - 2) * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][1] if a == 0]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var([q for (i, a, q) in history[attr_i][1] if a == 0]))}$")

	axs[1, (attr_i - 2) * 2].hist([q for (i, a, q) in history[attr_i][0] if a == 1], color="orange", bins=3, density=True)
	axs[1, (attr_i - 2) * 2].set_xlabel(f"$\Delta Q(s_{{i \leq {attr_i}}}, RIGHT)$")
	# axs[1, (attr_i - 2) * 2].set_xlim(-0.5, 0.5)
	axs[1, (attr_i - 2) * 2].set_title(f"$i \leq {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][0] if a == 1]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var([q for (i, a, q) in history[attr_i][0] if a == 1]))}$")

	axs[1, (attr_i - 2) * 2 + 1].hist([q for (i, a, q) in history[attr_i][1] if a == 1], color="orange", bins=3, density=True)
	axs[1, (attr_i - 2) * 2 + 1].set_xlabel(f"$\Delta Q(s_{{i > {attr_i}}}, RIGHT)$")
	# axs[1, (attr_i - 2) * 2 + 1].set_xlim(-0.5, 0.5)
	axs[1, (attr_i - 2) * 2 + 1].set_title(f"$i > {attr_i}$ \n $\mu = {'{:.4f}'.format(np.mean([q for (i, a, q) in history[attr_i][1] if a == 1]))}$ \n $\sigma^2 = {'{:.4f}'.format(np.var([q for (i, a, q) in history[attr_i][1] if a == 1]))}$")

plt.show()