import argparse
import pickle
import numpy as np

from imitlearn.distilled_tree import DistilledTree
from rich import print

import imitlearn.env_configs

class CartOvaAgent:
    def __init__(self, config):
        self.config = config
        self.actions = config['actions']
        self.n_actions = config['n_actions']
        self.n_attributes = config['n_attributes']
    
    def fit(self, X, y, pruning):
        self.trees = [None] * self.n_actions

        for action_id, action in enumerate(self.actions):
            y_action = [(1 if y_i == action_id else 0) for y_i in y]

            new_config = self.config.copy()
            new_config['n_actions'] = 2
            new_config['actions'] = ["- - -", action]

            tree = DistilledTree(new_config)
            tree.fit(X, y_action, pruning=pruning)
            
            self.trees[action_id] = tree
    
    def act(self, state):
        state = state.reshape(1, self.n_attributes)
        probs = [tree.model.predict_proba(state)[0][1] for tree in self.trees]
        best_action = np.argmax(probs)
        return best_action
    
    def get_size(self):
        return [tree.get_size() for tree in self.trees]
    
    def get_as_viztree(self):
        output = ""

        for i, tree in enumerate(self.trees):
            action = self.actions[i]

            output += f"=>=>=> TREE for action {action.upper()}"
            output += tree.get_as_viztree(show_prob=True)
            output += "\n\n"
        
        return output
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.trees, f)
        print(f"Saved OVA to '{filename}'.")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.trees = pickle.load(f)
    
    def prune_redundant_leaves(self):
        for tree in self.trees:
            tree.prune_redundant_leaves()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavior Cloning')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-d','--dataset', help='Number of iterations to run', required=True, type=int)
    parser.add_argument('-e','--expert', help='Which expert to load, if needed', required=False)
    parser.add_argument('-x','--expert_class', help='What is the expert\'s class', required=False)
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=-1, type=int)
    parser.add_argument('--grading_episodes', help='How many episodes should we use to measure model\'s accuracy?', required=False, default=100, type=int)
    parser.add_argument('--should_print_state', help='Should print state?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = env_configs.get_config(args['task'])
    