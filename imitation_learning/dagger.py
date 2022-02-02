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
from imitation_learning.utils import load_dataset, printv, save_dataset
from imitation_learning.distilled_tree import DistilledTree

PRUNING_PARAM = 0.01
DATASET_SIZE = 50000

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

    filename = "data/cartpole_nn_19"

    # config = {
    #     "name": "LunarLander-v2",
    #     "can_render": True,
    #     "n_actions": 4,
    #     "actions": ["nop", "left engine", "main engine", "right engine"],
    #     "n_attributes": 8,              
    #     "attributes": [
    #         ("X Position", "continuous", -1, -1),
    #         ("Y Position", "continuous", -1, -1),
    #         ("X Velocity", "continuous", -1, -1),
    #         ("Y Velocity", "continuous", -1, -1),
    #         ("Angle", "continuous", -1, -1),
    #         ("Angular Velocity", "continuous", -1, -1),
    #         ("Leg 1 is Touching", "binary", [0, 1], -1),
    #         ("Leg 2 is Touching", "binary", [0, 1], -1)],
    # }

    # filename = "data/lunarlander_nn_9"

    # avg_reward, rewards = get_average_reward(config, expert)
    # print(f"Average reward for the expert: {avg_reward} ± {np.std(rewards)}.")

    # Initialization
    expert = ann.MLPAgent(config, exploration_rate=0)
    expert.load_model(filename)

    best_reward = -9999
    best_model = None

    episodes = 100
    
    # X, y = get_dataset_from_model(config, expert, episodes)
    X, y = load_dataset(f"{filename}_dataset")
    dt = DistilledTree(config)
    dt.fit(X, y, pruning=PRUNING_PARAM)

    data = []

    for i in range(100):
        # Collect trajectories from student and correct them with expert
        X2, _ = get_dataset_from_model(config, dt, episodes)
        y2 = label_dataset_with_model(config, expert, X2)

        # Aggregate datasets
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))

        # Sample from dataset aggregation
        # D = list(zip(X, y))
        # D = random.sample(D, DATASET_SIZE)
        # X, y = zip(*D)

        # Train new student
        dt = DistilledTree(config)
        dt.fit(X, y, pruning=PRUNING_PARAM)

        # Housekeeping
        printv(f"Step #{i}.")
        avg_reward, rewards = get_average_reward(config, dt)
        deviation = np.std(rewards)
        leaves = dt.model.get_n_leaves()
        depth = dt.model.get_depth()

        printv(f"- Obtained tree with {leaves} leaves and depth {depth}.")
        printv(f"- Average reward for the student: {avg_reward} ± {deviation}.")

        data.append((i, avg_reward, deviation, leaves, depth))

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = dt
    
    iterations, avg_rewards, deviations, leaves, depths = zip(*data)

    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations, color="red", alpha=0.2)
    ax1.plot(iterations, avg_rewards, color="red")
    ax1.set_xlabel("Pruning $\\alpha$")
    ax1.set_ylabel("Average reward")
    ax2.plot(iterations, leaves, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Pruning $\\alpha$")
    plt.suptitle(f"DAgger for {config['name']} w/ pruning $\\alpha = {PRUNING_PARAM}$")
    plt.show()

    dt = best_model
    printv(f"Visualizing final tree:")
    dt.save_fig()
    dt.save_model(f"data/dagger_best_tree_{config['name']}")
    visualize_model(config, dt, 10)

    # qtree = dt.get_as_qtree()
    # qtree.sort(key = lambda x : x[0])
    # save_tree_from_print(
    #     qtree,
    #     ["left", "right"],
    #     "_altopt_cartpole")