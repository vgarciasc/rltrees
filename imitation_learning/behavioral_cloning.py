import gym
import random
import pdb
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree

import ann
from il import *
from imitation_learning.utils import printv

class DistilledTree:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y):
        clf = tree.DecisionTreeClassifier(ccp_alpha=0.05)
        clf = clf.fit(X, y)
        self.model = clf

    def act(self, state):
        state = state.reshape(1, -1)
        action = self.model.predict(state)
        action = action[0]
        return action
    
    def save_fig(self):
        plt.figure(figsize=(15, 15))
        feature_names = [name for (name, _, _, _) in config["attributes"]]
        tree.plot_tree(self.model, feature_names=feature_names)
        plt.savefig('last_tree.png')

def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, done, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y

def save_dataset(filename, X, y):
    with open(filename, "wb") as f:
        pickle.dump((X, y), f)

def load_dataset(filename):
    with open(filename, "rb") as f:
        X, y = pickle.load(f)
    return X, y

if __name__ == "__main__":
    config = {
        "name": "CartPole-v1",
        "can_render": True,
        "episode_max_score": 195,
        "should_force_episode_termination_score": True,
        "episode_termination_score": 0,
        "n_actions": 2,
        "actions": ["left", "right"],
        "n_attributes": 4,              
        "attributes": [
            ("Cart Position", "continuous", -1, -1),
            ("Cart Velocity", "continuous", -1, -1),
            ("Pole Angle", "continuous", -1, -1),
            ("Pole Angular Velocity", "continuous", -1, -1)],
    }

    filename = "data/cartpole_nn_7"

    model = ann.MLPAgent(config, exploration_rate=0)
    model.load_model(filename)

    X, y = get_dataset_from_model(config, model, 100)
    save_dataset(filename + "_dataset", X, y)
    print(f"Dataset size: {len(X)}")

    # dt = DistilledTree(config)
    # dt.fit(X, y)

    # dt.save_fig()

    # get_average_reward(config, dt, verbose=True)
    # visualize_model(config, dt, 10)    