import gym
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from sklearn import tree

import ann
import argparse
import imitation_learning.env_configs
import imitation_learning.parser
from il import *
from qtree import save_tree_from_print
from imitation_learning.utils import printv, load_dataset, save_dataset
from imitation_learning.distilled_tree import DistilledTree
from imitation_learning.keras_dnn import KerasDNN

def run_behavior_cloning(config, X, y, pruning_alpha):
    dt = DistilledTree(config)
    dt.fit(X, y, pruning=pruning_alpha)
    return dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--episodes_to_grade_model', help='How many episodes to grade model?', required=False, default=100, type=int)
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    expert, X, y = imitation_learning.parser.handle_args(args, config)
    
    # Train decision tree
    dt = run_behavior_cloning(config, X, y, args['pruning'])
    dt.save_fig()

    # Printing results
    avg, rewards = get_average_reward(
        config, dt,
        episodes=args['episodes_to_grade_model'],
        verbose=args['verbose'] if args['episodes_to_grade_model'] < 100 else False)
    deviation = np.std(rewards)
    print(f"Average reward is {avg} ± {deviation}.")
    print(f"Resulting tree has {dt.model.get_n_leaves()} leaves and depth {dt.model.get_depth()}.")

    # Saving results
    dt.save_model(f"data/best_bc_{config['name']}")
    qtree = dt.get_as_qtree()
    save_tree_from_print(
        qtree,
        config['actions'],
        f"_{config['name']}_bc_{args['pruning']}")
    
    # Visualizing model
    if args['should_visualize']:
        visualize_model(config, dt, 25)