import gym
import pickle
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree

import ann
import imitlearn.env_configs
from imitlearn.il import *
from qtree import save_tree_from_print
from imitlearn.utils import printv, load_dataset, save_dataset
from imitlearn.distilled_tree import DistilledTree
from imitlearn.keras_dnn import KerasDNN
from imitlearn.behavioral_cloning import run_behavior_cloning

def run_grid_behavior_cloning(config, X, y, start, end, steps, 
    episodes_to_grade=100, should_save_trees=False, verbose=False):
    history = []

    for i, pruning_alpha in enumerate(np.linspace(start, end, steps)):
        # Run behavior cloning for this value of pruning
        dt = run_behavior_cloning(
            config, X, y,
            pruning_alpha=pruning_alpha)

        # Evaluating tree
        avg_reward, rewards = get_average_reward(config, dt, episodes=episodes_to_grade)
        deviation = np.std(rewards)

        # Keeping history of trees
        leaves = dt.model.get_n_leaves()
        depth = dt.model.get_depth()
        history.append((pruning_alpha, avg_reward, deviation, leaves, depth))

        # Logging info if necessary
        printv(f"#({i} / {steps}) PRUNING = {pruning_alpha}: \t"
            + f"REWARD = {'{:.3f}'.format(avg_reward)} ± {'{:.3f}'.format(deviation)}"
            + f"\tNODES: {leaves*2 -1}, DEPTH: {depth}.",
            verbose)

        # Saving tree
        if should_save_trees:
            qtree = dt.get_as_qtree()
            qtree.sort(key = lambda x : x[0])
            save_tree_from_print(
                qtree, config['actions'],
                f"_{config['name']}_bc_pruning_{pruning_alpha}")
    
    # pruning_params, avg_rewards, deviations, leaves, depths = zip(*history)
    return zip(*history)

def plot_behavior_cloning(history):
    pruning_params, avg_rewards, deviations, leaves, depths = history
    
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
    plt.suptitle(f"Behavior cloning for {config['name']}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-s','--start', help='Starting point for pruning alpha', required=True, type=float)
    parser.add_argument('-e','--end', help='Ending point for pruning alpha', required=True, type=float)
    parser.add_argument('-i','--steps', help='Number of overall steps', required=True, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save_trees', help='Should save intermediary trees?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--episodes_to_grade_model', help='How many episodes to grade model?', required=False, default=100, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitlearn.env_configs.get_config(args['task'])
    expert, X, y = imitlearn.parser.handle_args(args, config)

    # Grid-running behavior cloning
    history = run_grid_behavior_cloning(
        config, X, y,
        start=args['start'],
        end=args['end'],
        steps=args['steps'],
        episodes_to_grade=args['episodes_to_grade_model'],
        should_save_trees=args['should_save_trees'],
        verbose=args['verbose'])

    # Plotting behavior cloning
    plot_behavior_cloning(history)
    