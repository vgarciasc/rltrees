import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree
from rich import print

total_rewards = []
env = gym.make("Catcher-v0")

for i_episode in range(10):
	state = env.reset()
	done = False
	reward = 0
	total_reward = 0
	t = 0

	while not done:
		t += 1

		env.render()

		action = 0
		state, reward, done, info = env.step(action)
		total_reward += reward

		if done:
			# print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
			total_rewards.append(total_reward)
			break
env.close()

print("Average reward per episode:", np.mean(total_rewards))