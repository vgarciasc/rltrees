import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from control_tree import CTNode, CTLeaf

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

ctree = CTNode((2, 0), 
			CTLeaf(0), 
			CTLeaf(1))

# Simple QTree
qtree = CTNode((2, 0),
			PHCTLeaf(),
			PHCTLeaf())

total_rewards = []
left_leaf_dataset = []

for i_episode in range(5000):
	state = env.reset()
	reward, total_reward = 0, 0
	done = False
	
	while not done:
		leaf, action = qtree.predict(state)

		if np.random.random() < (eps / (i_episode / 10 + 1)):
			action = np.random.randint(0, env.action_space.n)

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = qtree.predict(next_state)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])

		leaf.q_values[action] += delta_q
		leaf.history.append((leaf.q_values[0], leaf.q_values[1]))

		total_reward += reward
		state = next_state

	total_rewards.append(total_reward)
		
print("Average reward per episode:", np.mean(total_rewards))

# # Print Simple Tree
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(range(len(qtree.left.history)), [a_left for (a_left, a_right) in qtree.left.history], color="green")
# axs[0, 0].set_title("$Q(s_{PA \leq 0}; LEFT)$")
# axs[1, 0].plot(range(len(qtree.left.history)), [a_right for (a_left, a_right) in qtree.left.history], color="green")
# axs[1, 0].set_title("$Q(s_{PA \leq 0}; RIGHT)$")
# axs[0, 1].plot(range(len(qtree.right.history)), [a_left for (a_left, a_right) in qtree.right.history], color="red")
# axs[0, 1].set_title("$Q(s_{PA > 0}; LEFT)$")
# axs[1, 1].plot(range(len(qtree.right.history)), [a_right for (a_left, a_right) in qtree.right.history], color="red")
# axs[1, 1].set_title("$Q(s_{PA > 0}; RIGHT)$")
# plt.show()

for i_episode in range(5000):
	state = env.reset()
	reward = 0
	done = False
	
	while not done:
		leaf, action = qtree.predict(state)

		if leaf == qtree.left and np.random.random() < 0.5:
			action = np.random.randint(0, env.action_space.n)

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = qtree.predict(next_state)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])

		leaf.q_values[action] += delta_q
		leaf.history.append((leaf.q_values[0], leaf.q_values[1]))

		if leaf == qtree.left:
			left_leaf_dataset.append((state, action, delta_q))

		state = next_state

env.close()

# # Print State and Delta_Q distribution
fig, axs = plt.subplots(2, 2)
moveleft_states_distribution = [s for (s, a, q) in left_leaf_dataset if a == 0]
moveright_states_distribution = [s for (s, a, q) in left_leaf_dataset if a == 1]
moveleft_q_distribution = [q for (s, a, q) in left_leaf_dataset if a == 0]
moveright_q_distribution = [q for (s, a, q) in left_leaf_dataset if a == 1]
axs[0, 0].scatter([a0 for (a0, a1, a2, a3) in moveleft_states_distribution], moveleft_q_distribution, s=2, color="green", label="LEFT")
axs[0, 0].scatter([a0 for (a0, a1, a2, a3) in moveright_states_distribution], moveright_q_distribution, s=2, color="lime", label="RIGHT")
axs[0, 0].set_xlabel("Cart Position")
axs[0, 0].set_ylabel("$\Delta Q(s_{PA \leq 0}; LEFT)$")
axs[0, 0].legend()
axs[1, 0].scatter([a1 for (a0, a1, a2, a3) in moveleft_states_distribution], moveleft_q_distribution, s=2, color="blue", label="LEFT")
axs[1, 0].scatter([a1 for (a0, a1, a2, a3) in moveright_states_distribution], moveright_q_distribution, s=2, color="turquoise", label="RIGHT")
axs[1, 0].set_xlabel("Cart Velocity")
axs[1, 0].set_ylabel("$\Delta Q(s_{PA \leq 0}; LEFT)$")
axs[1, 0].legend()
axs[0, 1].scatter([a2 for (a0, a1, a2, a3) in moveleft_states_distribution], moveleft_q_distribution, s=2, color="orange", label="LEFT")
axs[0, 1].scatter([a2 for (a0, a1, a2, a3) in moveright_states_distribution], moveright_q_distribution, s=2, color="red", label="RIGHT")
axs[0, 1].set_xlabel("Pole Angle")
axs[0, 1].set_ylabel("$\Delta Q(s_{PA \leq 0}; LEFT)$")
axs[0, 1].legend()
axs[1, 1].scatter([a3 for (a0, a1, a2, a3) in moveleft_states_distribution], moveleft_q_distribution, s=2, color="purple", label="LEFT")
axs[1, 1].scatter([a3 for (a0, a1, a2, a3) in moveright_states_distribution], moveright_q_distribution, s=2, color="magenta", label="RIGHT")
axs[1, 1].legend()
axs[1, 1].set_xlabel("Pole Angular Velocity")
axs[1, 1].set_ylabel("$\Delta Q(s_{PA \leq 0}; LEFT)$")
plt.show()