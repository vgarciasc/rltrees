import gym
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree

import ann
from il import *
from qtree import save_tree_from_print
from imitation_learning.utils import printv, load_dataset, save_dataset
from imitation_learning.distilled_tree import DistilledTree

if __name__ == "__main__":
    # config = {
    #     "name": "CartPole-v1",
    #     "can_render": True,
    #     "episode_max_score": 195,
    #     "should_force_episode_termination_score": True,
    #     "episode_termination_score": 0,
    #     "n_actions": 2,
    #     "actions": ["left", "right"],
    #     "n_attributes": 4,              
    #     "attributes": [
    #         ("Cart Position", "continuous", -1, -1),
    #         ("Cart Velocity", "continuous", -1, -1),
    #         ("Pole Angle", "continuous", -1, -1),
    #         ("Pole Angular Velocity", "continuous", -1, -1)],
    # }

    # filename = "data/cartpole_nn_19"

    config = {
        "name": "LunarLander-v2",
        "can_render": True,
        "n_actions": 4,
        "actions": ["nop", "left engine", "main engine", "right engine"],
        "n_attributes": 8,              
        "attributes": [
            ("X Position", "continuous", -1, -1),
            ("Y Position", "continuous", -1, -1),
            ("X Velocity", "continuous", -1, -1),
            ("Y Velocity", "continuous", -1, -1),
            ("Angle", "continuous", -1, -1),
            ("Angular Velocity", "continuous", -1, -1),
            ("Leg 1 is Touching", "binary", [0, 1], -1),
            ("Leg 2 is Touching", "binary", [0, 1], -1)],
    }

    filename = "data/lunarlander_nn_9"

    model = ann.MLPAgent(config, exploration_rate=0)
    model.load_model(filename)

    print("== Neural Network")
    get_average_reward(config, model, episodes=100, verbose=True)

    # X, y = get_dataset_from_model(config, model, 300)
    # save_dataset(filename + "_dataset", X, y)
    # print(f"Dataset size: {len(X)}")

    # X, y = load_dataset(filename + "_dataset")
    # dt = DistilledTree(config)
    # dt.fit(X, y, pruning=0.00005)

    # dt.save_fig()
    # print(f"Resulting tree has {dt.model.get_n_leaves()} leaves and depth {dt.model.get_depth()}.")

    # print("== Decision Tree")
    # get_average_reward(config, dt, episodes=50, verbose=True)

    # # visualize_model(config, dt, 10)
    # qtree = dt.get_as_qtree()
    # qtree.sort(key = lambda x : x[0])
    # save_tree_from_print(
    #     qtree,
    #     ["nop", "left engine", "main engine", "right engine"],
    #     "_lunarlander_bc_pruning_00005")
    
    # # print(qtree)