import pdb
import numpy as np
import matplotlib.pyplot as plt

from imitlearn.ann import MLPAgent
from imitlearn.bj_tabular_q import BlackjackQLearner
from imitlearn.il import *
from imitlearn.utils import load_dataset, printv, save_dataset
from imitlearn.keras_dnn import KerasDNN

def handle_args(args, config):
    filename = args['expert_filepath']

    if args['expert_class'] == "KerasDNN":
        expert = KerasDNN(config, exploration_rate=args['expert_exploration_rate'])
        expert.load(filename)
    elif args['expert_class'] == "MLP":
        expert = MLPAgent(config, exploration_rate=args['expert_exploration_rate'])
        expert.load_model(filename)
    elif args['expert_class'] == "QTable" and args['task'] == "blackjack":
        expert = BlackjackQLearner(config, exploration_rate=args['expert_exploration_rate'])
        expert.load_model(filename)

    if args['should_grade_expert']:
        avg_reward, rewards = get_average_reward(config, expert, verbose=True)
        printv(f"Average reward for the expert: {avg_reward} ± {np.std(rewards)}.")
        printv("")

    if args['should_collect_dataset']:
        X, y = get_dataset_from_model(config, expert, args['dataset_size'], args['verbose'])
        save_dataset(f"{filename}_dataset", X, y)
        print(f"Create dataset with {len(y)} observations for {config['name']}.")
    else:
        X, y = load_dataset(f"{filename}_dataset")
        print(f"Dataset length: {len(y)}")
        
    return expert, X, y