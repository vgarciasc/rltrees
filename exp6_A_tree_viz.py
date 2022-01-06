import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree

qtree = None
with open('data/tree 2022-01-05 11-12_pruning', 'rb') as file:
	qtree = pickle.load(file)
	file.close()

qtree.print_tree()

total_rewards = []
env = gym.make("CartPole-v1")

for i_episode in range(20):
	observation = env.reset()
	total_reward = 0

	for t in range(1000):
		env.render()
		
		_, action = qtree.predict(observation)
		observation, reward, done, info = env.step(action)
		total_reward += reward

		if done:
			print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
			total_rewards.append(total_reward)
			break
env.close()

print("Average reward per episode:", np.mean(total_rewards))