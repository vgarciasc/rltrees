import argparse
import gym
import numpy as np

from distilled_tree import DistilledTree
from il import visualize_model
from imitation_learning.il import get_average_reward
from qtree import save_tree_from_print
from rich import print

import imitation_learning.env_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    # parser.add_argument('-c','--expert_class', help='Expert class is MLP or KerasDNN?', required=True)
    parser.add_argument('-i','--iterations', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    filename = args['filename']
    
    dt = DistilledTree(config)
    dt.load_model(filename)
    
    dt.save_fig()

    if args['should_visualize']:
        visualize_model(config, dt, args['iterations'])
    
    avg_reward, rewards = get_average_reward(config, dt, 100, verbose=args['verbose'])
    deviation = np.std(rewards)
    leaves = dt.model.get_n_leaves()
    depth = dt.model.get_depth()
    print(f"- Obtained tree with {leaves} leaves and depth {depth}.")
    print(f"- Average reward for the best policy: {avg_reward} ± {deviation}.")

    # qtree = dt.get_as_qtree()
    # save_tree_from_print(
    #     qtree,
    #     config['actions'],
    #     f"_dagger_best_tree_MountainCar-v0")