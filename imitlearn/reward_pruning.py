from imitlearn.utils import printv
import imitlearn.env_configs
import numpy as np
import gym
import pickle
import argparse
import pdb
import matplotlib.pyplot as plt

from rich import print
from qtree import QLeaf, QNode, save_tree

def get_average_reward_qtree(qtree, config, episodes=10, verbose=False):
    gym_env = gym.make(config['name'])

    rewards = np.zeros(episodes)
    for i in range(episodes):
        state = gym_env.reset()
        reward = 0
        done = False
        
        while not done:
            _, action = qtree.predict(state)
            next_state, reward, done, _ = gym_env.step(action)
            rewards[i] += reward
            state = next_state
    
    mean = np.mean(rewards)
    stdev = np.std(rewards)

    printv(f"Average reward: {mean} ± {stdev}", verbose)
    
    return np.mean(rewards), np.std(rewards)

def is_inner_node(node):
    return node.__class__.__name__ == "QNode"

def is_leaf(node):
    return node.__class__.__name__ == "QLeaf"

def prune_by_reward(qtree, node, config, 
    episodes_per_prune=100,
    comp_threshold = 0.95,
    verbose=False):

    params = (config, episodes_per_prune, comp_threshold, verbose)
    history = []

    if is_leaf(node):
        return qtree, []

    if is_inner_node(node.left):
        qtree, new_history = prune_by_reward(qtree, node.left, *params)
        history += new_history

    if is_inner_node(node.right):
        qtree, new_history = prune_by_reward(qtree, node.right, *params)
        history += new_history
    
    parent_of_leaves = is_leaf(node.left) and is_leaf(node.right)
    parent_of_subtree = is_inner_node(node.right) or is_inner_node(node.left)
    
    printv(f"\n-- Evaluating node {node.pretty_string(config)}...", verbose)
    if parent_of_leaves or parent_of_subtree:
        avg_reward, deviation = get_average_reward_qtree(qtree, config, episodes=episodes_per_prune)
        printv(f"\tThe average reward of the tree is {avg_reward} ± {deviation}.", verbose)
        changing_left_node = (node.parent and node == node.parent.left)
        
        if parent_of_leaves:
            merged_q_values = [node.left.q_values, node.right.q_values][np.argmax([np.max(node.left.q_values), np.max(node.right.q_values)])]
            merged_leaf = QLeaf(node.parent, node.parent is not None and node == node.parent.left, config['actions'], merged_q_values)

            if node.parent:
                if node == node.parent.left:
                    node.parent.left = merged_leaf
                else:
                    node.parent.right = merged_leaf
            else:
                qtree = merged_leaf

            printv(f"\tMerged leaves of node '{config['attributes'][node.attribute][0]} <= {node.value}'!", verbose)
        
        elif parent_of_subtree:
            if node.left.__class__.__name__ == "QNode":
                node.left.parent = node.parent
            else:
                node.right.parent = node.parent

            if node.parent:
                if node == node.parent.left:
                    node.parent.left = node.left if node.left.__class__.__name__ == "QNode" else node.right
                else:
                    node.parent.right = node.left if node.left.__class__.__name__ == "QNode" else node.right
            else:
                qtree = node.left if node.left.__class__.__name__ == "QNode" else node.right

            child_node = node.left if node.left.__class__.__name__ == "QNode" else node.right
            printv(f"\tRouted from node '{config['attributes'][node.attribute][0]} <= {node.value}' to its subtree '{config['attributes'][child_node.attribute][0]} <= {child_node.value}'!", verbose)

        new_avg_reward, new_deviation = get_average_reward_qtree(qtree, config, episodes=episodes_per_prune)
        printv(f"\tThe average reward of the tree is {new_avg_reward} ± {new_deviation}.", verbose)
        printv(f"\tGot average reward {new_avg_reward} after merge.", verbose)

        reward_comparison_threshold = (comp_threshold if avg_reward > 0 else (2 - comp_threshold))
        if new_avg_reward < reward_comparison_threshold * avg_reward:
            printv("\tAverage reward was reduced too much. Undoing merge...", verbose)
            if node.parent:
                if changing_left_node:
                    node.parent.left = node
                else:
                    node.parent.right = node
            else:
                qtree = node
            
            if parent_of_subtree:
                (node.left if node.left.__class__.__name__ == "QNode" else node.right).parent = node
        else:
            printv(f"\tAppending {(qtree.get_size(), new_avg_reward)}", verbose)
            history.append((qtree.get_size(), new_avg_reward, new_deviation))

    return qtree, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning by Rewards')
    parser.add_argument('-t','--task', help="Which task to run?", required=True)
    parser.add_argument('-f','--filepath', help='Filepath to load old tree', required=True)
    parser.add_argument('-o','--output_filepath', help='Filepath to save pruned tree', required=True)
    parser.add_argument('--comp_threshold', help='Reward comparison threshold', required=False, default=0.95, type=float)
    parser.add_argument('--max_pruning_iters', help='Maximum number of pruning iterations', required=False, default=5, type=int)
    parser.add_argument('--episodes_per_prune', help='Number of episodes used to evaluate pruning', required=False, default=100, type=int)
    parser.add_argument('--pruning_cycles', help='How many pruning cycles should be done?', required=False, default=1, type=int)
    parser.add_argument('--grading_episodes', help='How many episodes to use during grading of final model?', required=False, default=10000, type=int)
    parser.add_argument('--should_plot', help='Should plot pruning results?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = imitlearn.env_configs.get_config(args['task'])
    output = []

    print(f"Starting reward pruning for {config['name']}, loading file '{args['filepath']}'.")

    for cycle in range(args['pruning_cycles']):
        original_size = None
        print(f"\n\n===== Pruning cycle: {cycle} =====")
        
        with open(args['filepath'], "rb") as f:
            qtree = pickle.load(f)
            original_size = qtree.get_size()
            printv(f"Before pruning, tree had {original_size} nodes.")
        
        if args['verbose']:
            qtree.print_tree()

        count = 0
        history = []
        plot_history = []
        while count == 0 or len(history) > 0:
            qtree, history = prune_by_reward(
                qtree, qtree, config,
                verbose=args['verbose'],
                episodes_per_prune=args['episodes_per_prune'])
            plot_history += history

            count += 1
            if count > args['max_pruning_iters']:
                break
        
        if args['should_plot']:
            tree_sizes, avg_rewards, std_rewards = zip(*plot_history)

            avg_rewards = np.array(avg_rewards)
            std_rewards = np.array(std_rewards)

            plt.title(f"File: {args['filepath']} \n\n Performance during Reward Pruning for {config['name']}")
            plt.fill_between(tree_sizes, avg_rewards - std_rewards, avg_rewards + std_rewards, color="blue", alpha=0.2)
            plt.plot(tree_sizes, avg_rewards, color='blue')
            plt.xlabel("Tree size")
            plt.gca().invert_xaxis()
            plt.ylabel("Average reward")
            plt.show()

        if args['verbose']:
            qtree.print_tree()
        
        printv(f"Before pruning, tree had {original_size} nodes.")
        printv(f"After pruning, tree has {qtree.get_size()} nodes.")
        avg, stdev = get_average_reward_qtree(
            qtree, config,
            episodes=args['grading_episodes'],
            verbose=True)
        output.append((avg, stdev, qtree.get_size()))
        save_tree(qtree, f"_{config['name']}_pruned_{cycle}")
    
    averages, deviations, tree_sizes = zip(*output)
    printv(f"Averages: {averages}, avg: {np.mean(averages)} ± {np.std(averages)}")
    printv(f"Tree size: {tree_sizes}, avg: {np.mean(tree_sizes)} ± {np.std(tree_sizes)}")
