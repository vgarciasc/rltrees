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

# Simple QTree 1
# qtree = CTNode((2, 0),
# 			PHCTLeaf(),
# 			PHCTLeaf())

# Simple QTree 2
qtree = CTNode((2, 0),
			CTNode((2, -0.15),
				PHCTLeaf(),
				PHCTLeaf()),
			PHCTLeaf())

total_rewards = []

for i_episode in range(5000):
	state = env.reset()
	reward, total_reward = 0, 0
	done = False
	t = 0
	
	while not done:
		t += 1
		# env.render()

		leaf, action = qtree.predict(state)
		# action = ctree.predict(state)

		if np.random.random() < (eps / (i_episode / 10 + 1)):
			action = np.random.randint(0, env.action_space.n)

		next_state, reward, done, _ = env.step(action)
		next_leaf, next_action = qtree.predict(next_state)
		
		if not done:
			delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
		else:
			delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])
		
		# print(f"Q({'s_PA <= 0' if state[2] <= 0 else 's_PA  > 0'}, {'L' if action == 0 else 'R'}) += {learning_rate} * ({reward} + {discount_factor} * max(Q({'s_PA <= 0' if next_state[2] <= 0 else 's_PA > 0'}, a) - Q({'s_PA <= 0' if state[2] <= 0 else 's_PA > 0'}, {'L' if action == 0 else 'R'})")
		# print(f"                += {learning_rate} * ({reward} + {discount_factor} * {'%.2f' % np.max(next_leaf.q_values)} - {'%.2f' % leaf.q_values[action]})")
		# print(f"                += {delta_q}")
		# print(f"                = {leaf.q_values[action] + delta_q}")

		# if done:
		# 	print(f"EPISODE {i_episode} ENDED")
		# 	print("-" * 25)

		leaf.q_values[action] += delta_q
		leaf.history.append((leaf.q_values[0], leaf.q_values[1]))

		total_reward += reward
		state = next_state

	# print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
	total_rewards.append(total_reward)
		
env.close()
print("Average reward per episode:", np.mean(total_rewards))