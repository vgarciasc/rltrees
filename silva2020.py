import gym
import numpy as np
from control_tree import CTNode, CTLeaf

env = gym.make("CartPole-v1")
# Silva et al. 2020 final tree
ctree = CTNode((3, 0.44), 
			CTNode((2, 0.01), 
				CTLeaf(0), 
				CTLeaf(1)),
			CTNode((3, -0.3), 
				CTNode((2, -0.0), 
					CTLeaf(1),
					CTLeaf(0)), 
				CTNode((2, -0.41), 
					CTLeaf(0), 
					CTLeaf(1))))

# Simple tree 1
# ctree = CTNode((2, 0), 
# 			CTLeaf(0), 
# 			CTLeaf(1))

# Simple tree 2
# ctree = CTNode((3, 0.44), 
# 			CTLeaf(0), 
# 			CTLeaf(1))

total_rewards = []

for i_episode in range(20):
	observation = env.reset()
	total_reward = 0

	for t in range(1000):
		env.render()
		
		action = ctree.predict(observation)
		observation, reward, done, info = env.step(action)
		total_reward += reward

		if done:
			print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
			total_rewards.append(total_reward)
			break
env.close()

print("Average reward per episode:", np.mean(total_rewards))