import gym
import pickle
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rich import print

import ann
import imitation_learning.env_configs
import imitation_learning.my_parser
from il import *
from qtree import save_tree_from_print
from utils import load_dataset, printv, save_dataset, str_avg
from distilled_tree import DistilledTree
from keras_dnn import KerasDNN
from behavioral_cloning import get_model_to_train

def run_dagger(config, X, y, model_name, pruning_alpha, expert, 
    iterations, episodes, episodes_to_evaluate=10, verbose=False):

    best_reward = -9999
    best_model = None

    dt = get_model_to_train(config, model_name)
    dt.fit(X, y, pruning=pruning_alpha)

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
        dt = get_model_to_train(config, model_name)
        dt.fit(X, y, pruning=pruning_alpha)

        # Housekeeping
        printv(f"Step #{i}.", verbose)

        avg, std = get_average_reward_with_std(
            config, dt, episodes=episodes_to_evaluate)
        model_size = dt.get_size()

        printv(f"- Dataset length: {len(X)}")
        printv(f"- Obtained tree with {dt.get_size()} nodes.", verbose)
        printv(f"- Average reward for the student: {str_avg(avg, std)}.", verbose)

        history.append((i, avg, std, model_size))

        if avg > best_reward:
            best_reward = avg
            best_model = dt
    
    return best_model, best_reward, zip(*history)

def plot_dagger(config, avg_rewards, deviations, nodes, pruning_alpha, episodes, show=False):
    avg_rewards = np.array(avg_rewards)
    deviations = np.array(deviations)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations,
        color="red", alpha=0.2)
    ax1.plot(iterations, avg_rewards, color="red")
    ax1.set_ylabel("Average reward")
    ax1.set_xlabel("Iterations")
    ax2.plot(iterations, nodes, color="blue")
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
    parser.add_argument('-c','--class', help='Model to use', required=True)
    parser.add_argument('-e','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-j','--episodes', help='Number of episodes to collect every iteration', required=True, type=int)
    parser.add_argument('--episodes_to_evaluate', help='Number of episodes to run when evaluating best model', required=False, default=100, type=int)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--expert_exploration_rate', help='The epsilon to use during dataset collection', required=False, default=0.0, type=float)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_plot', help='Should plot performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(my_parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    expert, X, y = imitation_learning.my_parser.handle_args(args, config)
    
    printv(f"Running {args['class']} DAgger for {config['name']} with pruning = {args['pruning']}.")
    # Running DAgger
    dt, reward, history = run_dagger(
        config, X, y,
        pruning_alpha=args['pruning'],
        model_name=args['class'],
        expert=expert, 
        iterations=args['iterations'],
        episodes=args['episodes'],
        episodes_to_evaluate=args['episodes_to_evaluate'],
        verbose=args['verbose'])
    iterations, avg_rewards, deviations, model_sizes = history

    # Plotting results
    if args['should_plot']:
        plot_dagger(
            config=config,
            avg_rewards=avg_rewards,
            deviations=deviations,
            model_sizes=model_sizes,
            pruning_alpha=args['pruning'],
            episodes=args['episodes'],
            show=args['should_plot'])

    # Printing the best model
    avg, std = get_average_reward_with_std(config, dt, 1000)
    printv(f"- Obtained tree with {dt.get_size()} nodes.")
    printv(f"- Average reward for the best policy: {str_avg(avg, std)}.")

    # Visualizing the best model
    if args['should_visualize']:
        printv(f"Visualizing final tree:")
        visualize_model(config, dt, 10)

    # Saving best model
    dt.save_model(f"data/dagger_best_tree_{config['name']}")
    date = datetime.now().strftime("tree_%Y-%m-%d_%H-%M")
    filename = f"data/{config['name']}_{date}_dagger_{args['pruning']}"
    if args['class'] == "DistilledTree":
        qtree = dt.get_as_qtree()
        qtree.save(filename)
    elif args['class'] == "CartOva":
        dt.save_model(filename)