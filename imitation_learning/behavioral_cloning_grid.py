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

    data = []

    for pruning in np.linspace(0.0, 0.001, 100):
        X, y = load_dataset(filename + "_dataset")
        dt = DistilledTree(config)
        dt.fit(X, y, pruning=pruning)

        leaves = dt.model.get_n_leaves()
        depth = dt.model.get_depth()

        avg_reward, rewards = get_average_reward(config, dt, episodes=50)
        deviation = np.std(rewards)
        print(f"alpha = {pruning}: \tREWARD = {'{:.3f}'.format(avg_reward)} +- {'{:.3f}'.format(deviation)} \t{leaves} leaves and depth {depth}.")

        data.append((pruning, avg_reward, deviation, leaves, depth))

        qtree = dt.get_as_qtree()
        qtree.sort(key = lambda x : x[0])
        save_tree_from_print(
            qtree,
            ["nop", "left engine", "main engine", "right engine"],
            f"_cartpole_bc_pruning_{pruning}")
    
    pruning_params, avg_rewards, deviations, leaves, depths = zip(*data)

    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(pruning_params, avg_rewards - deviations, avg_rewards + deviations, color="red", alpha=0.2)
    ax1.plot(pruning_params, avg_rewards, color="red")
    ax1.set_xlabel("Pruning $\\alpha$")
    ax1.set_ylabel("Average reward")
    ax2.plot(pruning_params, leaves, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Pruning $\\alpha$")
    plt.show()