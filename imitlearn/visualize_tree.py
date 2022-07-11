import argparse
import gym
import pdb
import time
import numpy as np
from imitlearn.dataset_creation import get_model

from imitlearn.rulelists import Rulelist
from qtree import load_tree, QLeaf, QNode
from imitlearn.distilled_tree import DistilledTree
from imitlearn.il import visualize_model
from imitlearn.il import get_average_reward
from qtree import save_tree_from_print
from rich import print
from imitlearn.dt_structure_viz import viztree2qtree, load_viztree

import imitlearn.env_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    parser.add_argument('-c','--class', help='Tree is QTree, Distilled Tree, or Viztree?', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-e','--expert', help='Which expert to load, if needed', required=False)
    parser.add_argument('-x','--expert_class', help='What is the expert\'s class', required=False)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=-1, type=int)
    parser.add_argument('--grading_episodes', help='How many episodes should we use to measure model\'s accuracy?', required=False, default=100, type=int)
    parser.add_argument('--should_print_state', help='Should print state?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitlearn.env_configs.get_config(args['task'])
    
    expert = None
    if args['expert']:
        expert = get_model(args['expert_class'], args['expert'], config)
    
    model = get_model(args['class'], args['filename'], config, expert)

    if args['should_visualize']:
        visualize_model(config, model,
            args['iterations'],
            args['should_print_state'])

    start_time = time.time()
    
    avg_reward, rewards = get_average_reward(
        config, model,
        episodes=args['grading_episodes'],
        verbose=args['verbose'])
    deviation = np.std(rewards)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds.")

    tree_size = None
    if args['class'] == "DistilledTree":
        tree_size = model.model.get_n_leaves() * 2 + 1
        depth = model.model.get_depth()
    elif args['class'] in ["QTree", "VizTree"]:
        tree_size = model.get_size()
        depth = model.get_depth()
    if tree_size:
        print(f"Tree has {tree_size} nodes and depth {depth}.")

    if args['task_solution_threshold'] != -1:
        solved_episodes = len([r for r in rewards if r >= args['task_solution_threshold']])
        print(f"Success in {solved_episodes} / {args['grading_episodes']} episodes ({'{:3f}'.format(solved_episodes / args['grading_episodes'])} %)")

    # qtree = dt.get_as_qtree()
    # save_tree_from_print(
    #     qtree,
    #     config['actions'],
    #     f"_dagger_best_tree_MountainCar-v0")