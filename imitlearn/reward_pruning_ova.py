from imitlearn.ova import CartOvaAgent
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

def get_average_reward(config, model, episodes=10, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if config['should_convert_state_to_array']:
                state = np.array(state)
            
            probs = []
            for tree in model.trees:
                values = tree.predict(state)[0].q_values
                values /= np.sum(values)
                probs.append(values[1])
            action = np.argmax(probs)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
        total_rewards.append(total_reward)
    
    env.close()
    
    average_reward = np.mean(total_rewards)
    printv(f"Average reward for this model is {'{:.3f}'.format(average_reward)} ± {'{:.3f}'.format(np.std(total_rewards))}.", verbose)

    return average_reward, total_rewards

def get_average_reward_with_std(config, model, episodes=10, verbose=False):
    avg_reward, rewards = get_average_reward(config, model, episodes, verbose)
    return avg_reward, np.std(rewards)

def is_inner_node(node):
    return node.__class__.__name__ == "QNode"

def is_leaf(node):
    return node.__class__.__name__ == "QLeaf"

def prune_by_reward(ova, qtree, tree_id,
    node, config, 
    episodes_per_prune=100,
    comp_threshold = 0.95,
    verbose=False):

    params = (config, episodes_per_prune, comp_threshold, verbose)
    history = []

    if is_leaf(node):
        return qtree, []

    if is_inner_node(node.left):
        qtree, new_history = prune_by_reward(ova, qtree, tree_id, node.left, *params)
        history += new_history

    if is_inner_node(node.right):
        qtree, new_history = prune_by_reward(ova, qtree, tree_id, node.right, *params)
        history += new_history
    
    parent_of_leaves = is_leaf(node.left) and is_leaf(node.right)
    parent_of_subtree = is_inner_node(node.right) or is_inner_node(node.left)
    
    printv(f"\n-- Evaluating node {node.pretty_string(config)}...", verbose)
    if parent_of_leaves or parent_of_subtree:
        avg_reward, deviation = get_average_reward_with_std(config, ova, episodes=episodes_per_prune)
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

        ova.trees[tree_id] = qtree
        new_avg_reward, new_deviation = get_average_reward_with_std(config, ova, episodes=episodes_per_prune)
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
            
            ova.trees[tree_id] = qtree
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
    parser.add_argument('--task_solution_threshold', help='Minimum reward to solve task', required=False, default=-1, type=int)
    # parser.add_argument('--should_plot', help='Should plot pruning results?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    # parser.add_argument('--should_visualize', help='Should visualize final tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = imitlearn.env_configs.get_config(args['task'])
    output = []

    print(f"Starting reward pruning for {config['name']}, loading file '{args['filepath']}'.")
        
    ova_model = CartOvaAgent(config)
    ova_model.load_model(args['filepath'])
    ova_model.trees = [tree.get_as_qtree() for tree in ova_model.trees]

    alt_configs = []
    for action in config['actions']:
        new_config = config.copy()
        new_config['n_actions'] = 2
        new_config['actions'] = ["- - -", action]
        alt_configs.append(new_config)

    for cycle in range(args['pruning_cycles']):
        print(f"\n\n===== Pruning cycle: {cycle + 1} =====")
        printv(f"Before pruning, OVA model had {ova_model.get_size()} nodes.")

        count, history, plot_history = 0, [], []
        while count == 0 or len(history) > 0:
            tree_id = count % len(ova_model.trees)
            qtree = ova_model.trees[tree_id]

            print("[green]Tree before:[/green]:")
            qtree.print_tree()
            qtree, history = prune_by_reward(
                ova_model, qtree, tree_id,
                qtree, alt_configs[tree_id],
                verbose=args['verbose'],
                comp_threshold=args['comp_threshold'],
                episodes_per_prune=args['episodes_per_prune'])
            print("\n\n[green]Tree after:[/green]:")
            qtree.print_tree()
            
            ova_model.trees[tree_id] = qtree
            plot_history += history
            count += 1

            if count > args['max_pruning_iters']:
                break

        printv(f"After pruning, model has {ova_model.get_size()} nodes.")
        
        printv(f"Evaluating performance of pruned model...")
        avg, stdev = get_average_reward_with_std(
            config, ova_model,
            episodes=args['grading_episodes'],
            verbose=True)
        output.append((avg, stdev, ova_model.get_size()))
    
    averages, deviations, tree_sizes = zip(*output)
    printv(f"Averages: {averages}, avg: {np.mean(averages)} ± {np.std(averages)}")
    printv(f"Tree size: {tree_sizes}, avg: {np.mean(tree_sizes)} ± {np.std(tree_sizes)}")

    printv(f"Final tree size: {ova_model.get_size()}.")
    _, total_rewards = get_average_reward(config, ova_model,
        episodes=args['grading_episodes'],
        verbose=args['verbose'])
    
    successes = len([reward for reward in total_rewards if reward > args['task_solution_threshold']])
    printv(f"Success rate is: {'{:2f}'.format(successes * 100)}.")

    pdb.set_trace()
