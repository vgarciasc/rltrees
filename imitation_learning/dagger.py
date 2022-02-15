import gym
import pickle
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich import print

import ann
import imitation_learning.env_configs
import imitation_learning.parser
from il import *
from qtree import save_tree_from_print
from imitation_learning.utils import load_dataset, printv, save_dataset
from imitation_learning.distilled_tree import DistilledTree
from imitation_learning.keras_dnn import KerasDNN

def run_dagger(config, X, y, pruning_alpha, expert, 
    iterations, episodes, episodes_to_evaluate=10, verbose=False):

    best_reward = -9999
    best_model = None

    dt = DistilledTree(config)
    dt.fit(X, y, pruning=args['pruning'])

    history = []

    for i in range(iterations):
        # Collect trajectories from student and correct them with expert
        X2, _ = get_dataset_from_model(config, dt, episodes)
        y2 = label_dataset_with_model(config, expert, X2)

        # Aggregate datasets
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))

        # Sample from dataset aggregation
        # D = list(zip(X, y))
        # D = random.sample(D, args['dataset_size'])
        # X, y = zip(*D)

        # Train new student
        dt = DistilledTree(config)
        dt.fit(X, y, pruning=pruning_alpha)

        # Housekeeping
        printv(f"Step #{i}.", verbose)
        avg_reward, rewards = get_average_reward(
            config, dt, episodes=episodes_to_evaluate)
        deviation = np.std(rewards)
        leaves = dt.model.get_n_leaves()
        depth = dt.model.get_depth()

        printv(f"- Dataset length: {len(X)}")
        printv(f"- Obtained tree with {leaves * 2 - 1} nodes and depth {depth}.", verbose)
        printv(f"- Average reward for the student: {avg_reward} ± {deviation}.", verbose)

        history.append((i, avg_reward, deviation, leaves, depth))

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = dt
    
    return best_model, best_reward, zip(*history)

def plot_dagger(config, avg_rewards, deviations, pruning_alpha, episodes, show=False):
    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations,
        color="red", alpha=0.2)
    ax1.plot(iterations, avg_rewards, color="red")
    ax1.set_ylabel("Average reward")
    ax1.set_xlabel("Iterations")
    ax2.plot(iterations, leaves, color="blue")
    ax2.set_ylabel("Number of leaves")
    ax2.set_xlabel("Iterations")
    plt.suptitle(f"DAgger for {config['name']} w/ pruning $\\alpha = {pruning_alpha}$" +
        f", {episodes} per iteration")
    
    if show:
        plt.show()
    else:
        plt.savefig(f"figures/dagger_{config['name']}_{pruning_alpha}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-e','--episodes', help='Number of episodes to collect every iteration', required=True, type=int)
    parser.add_argument('--episodes_to_evaluate', help='Number of episodes to run when evaluating best model', required=False, default=100, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--expert_exploration_rate', help='The epsilon to use during dataset collection', required=False, default=0.0, type=float)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    expert, X, y = imitation_learning.parser.handle_args(args, config)
    
    printv(f"Running DAgger for {config['name']} with pruning level alpha = {args['pruning']}.")
    # Running DAgger
    dt, reward, history = run_dagger(
        config, X, y,
        pruning_alpha=args['pruning'],
        expert=expert, 
        iterations=args['iterations'],
        episodes=args['episodes'],
        episodes_to_evaluate=args['episodes_to_evaluate'],
        verbose=args['verbose'])
    iterations, avg_rewards, deviations, leaves, depths = history

    # Plotting results
    plot_dagger(
        config=config,
        avg_rewards=avg_rewards,
        deviations=deviations,
        pruning_alpha=args['pruning'],
        episodes=args['episodes'],
        show=args['should_plot']
    )

    # Printing the best model
    avg_reward, rewards = get_average_reward(config, dt, 1000)
    deviation = np.std(rewards)
    leaves = dt.model.get_n_leaves()
    depth = dt.model.get_depth()
    printv(f"- Obtained tree with {leaves * 2 - 1} nodes and depth {depth}.")
    printv(f"- Average reward for the best policy: {avg_reward} ± {deviation}.")

    # Visualizing the best model
    dt.save_fig()
    dt.save_model(f"data/dagger_best_tree_{config['name']}")
    if args['should_visualize']:
        printv(f"Visualizing final tree:")
        visualize_model(config, dt, 10)

    # Saving ebst model as QTree
    qtree = dt.get_as_qtree()
    save_tree_from_print(
        qtree,
        config['actions'],
        f"_dagger_{config['name']}")