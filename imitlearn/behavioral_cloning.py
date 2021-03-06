import gym
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from rich import print
from sklearn import tree

import argparse
from qtree import save_tree_from_print
import imitlearn.ann
import imitlearn.env_configs
import imitlearn.parser
from imitlearn.ova import CartOvaAgent
from imitlearn.il import *
from imitlearn.utils import printv, load_dataset, save_dataset
from imitlearn.distilled_tree import DistilledTree
from imitlearn.keras_dnn import KerasDNN
# from imitlearn.tnt_wrapper import TnTWrapper

def get_model_to_train(config, name):
    if name == "DistilledTree":
        return DistilledTree(config)
    elif name == "CartOva":
        return CartOvaAgent(config)
    # elif name == "TnT":
    #     return TnTWrapper(config)
    return None

def run_behavior_cloning(config, X, y, model_name, pruning_alpha):
    dt = get_model_to_train(config, model_name)
    dt.fit(X, y, pruning=pruning_alpha)
    return dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-c','--class', help='Model to use', required=True)
    parser.add_argument('-e','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-f','--expert_filepath', help='Filepath for expert', required=True)
    parser.add_argument('-p','--pruning', help='Pruning alpha to use', required=True, type=float)
    parser.add_argument('--should_collect_dataset', help='Should collect and save new dataset?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset_size', help='Size of new dataset to create', required=False, default=0, type=int)
    parser.add_argument('--expert_exploration_rate', help='The epsilon to use during dataset collection', required=False, default=0.0, type=float)
    parser.add_argument('--should_grade_expert', help='Should collect expert\'s metrics?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--grading_episodes', help='How many episodes to grade model?', required=False, default=100, type=int)
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitlearn.env_configs.get_config(args['task'])
    expert, X, y = imitlearn.parser.handle_args(args, config)
    
    # Train decision tree
    print("")
    print("==> Now training...")
    start_time = time.time()
    dt = run_behavior_cloning(config, X, y, args['class'], args['pruning'])
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds.")
    # dt.save_fig()

    print(f"Tree has size: {dt.get_size()}")

    # Printing results
    print("")
    print("==> Measuring average reward...")
    avg, rewards = get_average_reward(
        config, dt,
        episodes=args['grading_episodes'],
        verbose=args['verbose'] if args['grading_episodes'] <= 100 else False)
    deviation = np.std(rewards)
    print(f"Average reward is {avg} ?? {deviation}.")

    # Saving results
    if args['class'] == "DistilledTree":
        print(f"Resulting tree has {dt.model.get_n_leaves()} leaves and depth {dt.model.get_depth()}.")

        # dt.save_model(f"data/best_bc_{config['name']}")
        qtree = dt.get_as_qtree()
        date = datetime.now().strftime("tree_%Y-%m-%d_%H-%M")
        qtree.save(f"data/{config['name']}_{date}_bc_{args['pruning']}")
        print(dt.get_as_viztree())
    elif args['class'] == "CartOva":
        print(dt.get_as_viztree())
        print("")
        print(f"Tree sizes are: {[tree.get_size() for tree in dt.trees]}")
    elif args['class'] == "TnT":
        date = datetime.now().strftime("tree_%Y-%m-%d_%H-%M")
        dt.save_model(f"data/{date}_tnt_{config['task']}")
    
    # Visualizing model
    if args['should_visualize']:
        visualize_model(config, dt, 25)