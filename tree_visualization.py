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

def view_tree_in_action(qtree, envname, episodes=5, render=True, verbose=True):
    qtree.print_tree()
    total_rewards = []
    gym_env = gym.make(envname)

    for _ in range(episodes):
        state = gym_env.reset()
        done = False
        reward = 0
        total_reward = 0
        t = 0

        while not done:
            t += 1
            if render:
                gym_env.render()

            _, action = qtree.predict(state)

            state, reward, done, _ = gym_env.step(action)
            total_reward += reward

            if done:
                if verbose:
                    print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
                total_rewards.append(total_reward)
                break
    gym_env.close()

    print("Average reward per episode:", np.mean(total_rewards))

if __name__ == "__main__":
    filename = "data/tree 2022-01-11 16-27"
    envname = "LunarLander-v2"

    qtree = None
    with open(filename, 'rb') as file:
        qtree = pickle.load(file)
        file.close()
    
    view_tree_in_action(qtree, envname, episodes=50)