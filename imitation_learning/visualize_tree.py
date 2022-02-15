import argparse
import gym
import pdb
import time
import numpy as np

from qtree import load_tree, QLeaf, QNode
from distilled_tree import DistilledTree
from il import visualize_model
from imitation_learning.il import get_average_reward
from qtree import save_tree_from_print
from rich import print

import imitation_learning.env_configs

def load_viztree(filename):
    with open(filename, "r") as f:
        return f.read()
    
def viztree2qtree(config, string):
    actions = [a.lower() for a in config['actions']]
    attributes = [name.lower() for name, _, _, _ in config['attributes']]

    lines = string.split("\n")

    parents = [None for _ in lines]
    child_count = [0 for _ in lines]

    for line in lines:
        depth = line.rindex("- ") + 1
        content = line[depth:].strip()

        parent = parents[depth - 1] if depth > 1 else None
        is_left = (child_count[depth - 1] == 0) if depth > 1 else None
        
        is_leaf = content.lower() in actions

        if not is_leaf:
            attribute, threshold = content.split(" <= ")
            
            attribute = attributes.index(attribute.lower())
            threshold = float(threshold)
            split = (attribute, threshold)

            node = QNode(split, parent)
        if is_leaf:
            action = actions.index(content.lower())

            q_values = np.zeros(len(actions))
            q_values[action] = 1

            node = QLeaf(parent, is_left, actions, q_values)
        
        if parent:
            if is_left:
                parent.left = node
            else:
                parent.right = node
        else:
            root = node

        parents[depth] = node
        child_count[depth] = 0
        child_count[depth - 1] += 1
    
    return root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    parser.add_argument('-c','--tree_class', help='Tree is QTree, Distilled Tree, or Viztree?', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
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

    # qtree = dt.get_as_qtree()
    # save_tree_from_print(
    #     qtree,
    #     config['actions'],
    #     f"_dagger_best_tree_MountainCar-v0")