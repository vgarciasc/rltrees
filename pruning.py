import gym
import copy
import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.stats as stats
from qtree import QNode, QLeaf, save_tree
from rich import print
from tree_visualization import view_tree_in_action

from exp7_clean import *

if __name__ == "__main__":
    config = {
		"name": "MountainCar-v0",
        "can_render": True,
        "episode_max_score": 195,
        "should_force_episode_termination_score": False,
        "episode_termination_score": 0,
        "n_actions": 3,
        "should_stop_if_no_splits": False,
        "max_iters_without_split": 3,
        "actions": ["left", "nop", "right"],
        "n_attributes": 2,              
        "attributes": [("Car Position", "continuous", -1, -1),
                       ("Car Velocity", "continuous", -1, -1)],

        "learning_rate": 0.05,
        "discount_factor": 0.95,
        "epsilon": 0.0,
        "continuous_quantiles": 5,
        "splitting_criterion": 'random',

        "cycle_length": 2,
        "nodes_to_grow": 0, 
        "collection_episodes": 0,
        "reward_estimation_episodes": 10,
        "qlearning_episodes": 0,
        "qlearning_episodes_after_growing": 0,

        "should_store_history": False,
        "history_storage_length": 0,
        "should_qlearn_inplace": False,
        "inherit_q_values_upon_split": False,
        "inherit_history_upon_split": False,
        "learning_method": None,
    }

    summary_reward = []
    summary_episodes_run = []
    trees = []

    with open("data/tree 2022-02-03 16-46_dagger_best_tree_MountainCar-v0", "rb") as f:
        qtree = pickle.load(f)

    for _ in range(1):
        episodes_run = 0
        qtree, history = run_pruned_CUT(config, qtree)
        trees.append(copy.deepcopy(qtree))
        summary_episodes_run.append(episodes_run)
        summary_reward.append(get_average_reward(qtree, config, 1000))

    for tree, episodes, reward in zip(trees, summary_episodes_run, summary_reward):
        print("\n")
        tree.print_tree()
        save_tree(tree)
        print(f"Reward: {reward}")
        print(f"Episodes run: {episodes}")

    print(f"Average of episode rewards: {np.mean(summary_reward)}")
    print(f"Average of episodes run: {np.mean(summary_episodes_run)}")
    print(f"Summary reward: {summary_reward}")
    print(f"Summary episodes run: {summary_episodes_run}")
