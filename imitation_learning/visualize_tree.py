import argparse
import gym
import pdb
import time
import numpy as np

from rulelists import Rulelist
from qtree import load_tree, QLeaf, QNode
from distilled_tree import DistilledTree
from il import visualize_model
from imitation_learning.il import get_average_reward
from qtree import save_tree_from_print
from rich import print
from imitation_learning.dt_structure_viz import viztree2qtree, load_viztree

import imitation_learning.env_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    parser.add_argument('-c','--tree_class', help='Tree is QTree, Distilled Tree, or Viztree?', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=-1, type=int)
    parser.add_argument('--grading_episodes', help='How many episodes should we use to measure model\'s accuracy?', required=False, default=100, type=int)
    parser.add_argument('--should_print_state', help='Should print state?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    filename = args['filename']

    if args['tree_class'] == "DistilledTree":
        dt = DistilledTree(config)
        dt.load_model(filename)
        dt.save_fig()
    elif args['tree_class'] == "QTree":
        dt = load_tree(filename)
    elif args['tree_class'] == "VizTree":
        string = load_viztree(filename)
        dt = viztree2qtree(config, string)
    elif args['tree_class'] == "Rulelist":
        dt = Rulelist(config)
        dt.load_txt(filename)

    if args['should_visualize']:
        visualize_model(config, dt,
            args['iterations'],
            args['should_print_state'])

    start_time = time.time()
    
    avg_reward, rewards = get_average_reward(
        config, dt,
        episodes=args['grading_episodes'],
        verbose=args['verbose'])
    deviation = np.std(rewards)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds.")

    if args['tree_class'] == "DistilledTree":
        tree_size = dt.model.get_n_leaves() * 2 + 1
        depth = dt.model.get_depth()
    elif args['tree_class'] in ["QTree", "VizTree"]:
        tree_size = dt.get_size()
        depth = dt.get_depth()
    print(f"Tree has {tree_size} nodes and depth {depth}.")

    if args['task_solution_threshold'] != -1:
        solved_episodes = len([r for r in rewards if r >= args['task_solution_threshold']])
        print(f"Success in {solved_episodes} / {args['grading_episodes']} episodes ({'{:3f}'.format(solved_episodes / args['grading_episodes'])} %)")

    # qtree = dt.get_as_qtree()
    # save_tree_from_print(
    #     qtree,
    #     config['actions'],
    #     f"_dagger_best_tree_MountainCar-v0")