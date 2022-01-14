import gym
import copy
import pdb
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.stats as stats
from qtree import QNode, QLeaf, save_tree, grow_tree
from rich import print
from tree_visualization import view_tree_in_action

episodes_run = 0


def grow_tree(tree, leaf, splitting_criterion, split=None):
    if split is None:
        split = splitting_criterion(leaf)

    new_node = QNode(split, leaf.parent, None, None)
    new_node.left = QLeaf(parent=new_node, is_left=True, actions=leaf.actions)
    # new_node.left.full_q_history = copy.deepcopy(leaf.full_q_history)
    new_node.right = QLeaf(parent=new_node, is_left=False, actions=leaf.actions)
    # new_node.right.full_q_history = copy.deepcopy(leaf.full_q_history)

    if leaf.parent is None:
        return new_node

    if leaf.is_left:
        leaf.parent.left = new_node
    else:
        leaf.parent.right = new_node
    
    return tree

def run_episodes(qtree, env, n_episodes, should_qlearn, should_store_history):
    global episodes_run
    gym_env = gym.make(env['name'])
    alpha = env['learning_rate']
    gamma = env['discount_factor']

    for _ in range(1, n_episodes):
        episodes_run += 1

        state = gym_env.reset()
        leaf, action = qtree.predict(state)
        reward = 0
        done = False
        
        while not done:
            if np.random.random() < 0.5:
                action = np.random.randint(0, env['n_actions'])

            next_state, reward, done, _ = gym_env.step(action)
            if done and env['should_force_episode_termination_score']:
                reward = env['episode_termination_score']
            next_leaf, next_action = qtree.predict(next_state)

            if should_qlearn:
                next_q = 0 if done else np.max(next_leaf.q_values)
                delta_q = alpha * (reward + gamma * next_q - leaf.q_values[action])
                leaf.q_values[action] += delta_q

            leaf.q_history[action].append((state, action, next_leaf.value, reward))
            leaf.state_history.append(state)

            if should_store_history and len(leaf.q_history[action]) > env['history_storage_length']:
                leaf.q_history[action].pop(0)
                leaf.state_history.pop(0)

            leaf = next_leaf
            state = next_state
            action = next_action
    gym_env.close()
    
    return qtree

def collect_data(qtree, env):
    return run_episodes(qtree, env, 
        env['collection_episodes'],
        env['should_qlearn_inplace'],
        env['should_store_history'])

def update_datapoints(node, env):
    if node.__class__.__name__ == "QLeaf":
        leaf = node
        
        for action_id in range(env['n_actions']):
            for (s, a, v, r) in leaf.q_history[action_id]:
                q = r + env['discount_factor'] * v
                leaf.full_q_history[a].append((s, a, q))
    else:
        if node.left is not None:
            update_datapoints(node.left, env)
        if node.right is not None:
            update_datapoints(node.right, env)

    return node

def is_score_better(score1, score2, env):
    if env['splitting_criterion'] == 'ks':
        distance1, p_value1 = score1
        distance2, p_value2 = score2
        
        equal_p = (np.abs(p_value1 - p_value2) < 0.01)
        p1_smaller_p2 = (p_value1 < p_value2 - 0.01)

        if p1_smaller_p2 or (equal_p and distance1 < distance2):
            return True
        
        return False
    elif env['splitting_criterion'] == 'variance':
        return score1[0] < score2[0]
    elif env['splitting_criterion'] == 'random':
        return score1[0] == 999

def get_cutoffs(leaf, env, attr_idx, attr):
    _, attr_type, start_value, end_value = attr

    if attr_type == "continuous":
        if len(leaf.state_history) < 100:
            return []
        return np.quantile([s[attr_idx] for s in leaf.state_history],
            np.linspace(0.1, 0.9, env['continuous_quantiles']))
    elif attr_type == "discrete":
        return range(start_value, end_value)
    elif attr_type == "categorical":
        return start_value
    elif attr_type == "binary":
        return [0]
    
    print("This should not be happening.")
    return []

def select_split(qtree, node, env, verbose=False):
    if node.__class__.__name__ == "QLeaf":
        leaf = node
        best_split = None
        best_score = [0, 1] if env['splitting_criterion'] == 'ks' else [np.inf, 0]

        if verbose:
            print(f"\n{'Left' if leaf.is_left else 'Right'} leaf{(' of x[' + str(leaf.parent.attribute) + '] <= ' + str(leaf.parent.value)) if leaf.parent is not None else ''}:")

        if env['splitting_criterion'] == 'random':
            attr_i = random.randint(0, env['n_attributes'])
            attr = env['attributes'][attr_i]
            if leaf.state_history:
                cutoff = random.choice([s[attr_i] for s in leaf.state_history])
                return leaf, (attr_i, cutoff), [999, None]

        for i, attr in enumerate(env['attributes']):
            attr_name, attr_type, start_value, end_value = attr
            for cutoff in get_cutoffs(leaf, env, i, attr):
                score = [0, 1]
                if env['splitting_criterion'] == 'variance':
                    weights = [len(leaf.full_q_history[a]) for a in range(env['n_actions'])] / np.sum([len(leaf.full_q_history[a]) for a in range(env['n_actions'])])
                
                for action in range(env['n_actions']):
                    L_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[i] <= cutoff]
                    R_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[i] > cutoff]

                    if L_partition and R_partition:
                        if env['splitting_criterion'] == 'variance':
                            p_right = len(R_partition) / (len(R_partition) + len(L_partition))
                            p_left = len(L_partition) / (len(R_partition) + len(L_partition))
                            
                            var = np.var(L_partition) * p_left + np.var(R_partition) * p_right
                            score[0] += weights[action] * var
                        
                        elif env['splitting_criterion'] == 'ks':
                            kstest = stats.ks_2samp(L_partition, R_partition)
                            score[0] += kstest[0]
                            score[1] *= kstest[1]
                        
                if verbose:
                    print(f"> Split {(attr_name, cutoff)} has score {score}")

                if is_score_better(score, best_score, env):
                    best_split = (i, cutoff)
                    best_score = score
            
        return leaf, best_split, best_score
    else:
        if node.left:
            left_leaf, left_best_split, left_best_score = select_split(qtree, node.left, env, verbose)
        if node.right:
            right_leaf, right_best_split, right_best_score = select_split(qtree, node.right, env, verbose)

        if is_score_better(left_best_score, right_best_score, env):
            return left_leaf, left_best_split, left_best_score
        else:
            return right_leaf, right_best_split, right_best_score

def run_qlearning(qtree, env):
    return run_episodes(qtree, env, 
        env['qlearning_episodes'],
        True, False)

def run_monte_carlo_control(qtree, env):
    gym_env = gym.make(env['name'])

    for T in range(1, env['qlearning_episodes']):
        global episodes_run; episodes_run += 1

        episode = []
        state = gym_env.reset()
        done = False

        while not done:
            leaf, action = qtree.predict(state)

            if np.random.random() < 0.5 * env['qlearning_episodes'] / T:
                action = np.random.randint(0, env['n_actions'])
            
            next_state, reward, done, _ = gym_env.step(action)
            # if done:
            # 	reward = 0
            episode.append((leaf, action, reward))
            state = next_state

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            leaf, action, reward = episode[t]
            
            G = env['discount_factor'] * G + reward

            has_appeared = False
            for i in range(0, t):
                leaf2, action2, _ = episode[i]
                if leaf == leaf2 and action == action2:
                    has_appeared = True
                    break
            
            if not has_appeared:
                leaf.q_history[action].append(G)
                leaf.q_values[action] = np.mean(leaf.q_history[action])
        
        qtree.reset_history()
    return qtree

def update_value(node):
    if node.__class__.__name__ == "QLeaf":
        leaf = node
        leaf.value = np.max(leaf.q_values)
    else:
        if node.left is not None:
            update_value(node.left)
        if node.right is not None:
            update_value(node.right)

    return node

def get_average_reward(qtree, env, episodes=None):
    global episodes_run

    gym_env = gym.make(env['name'])
    n_episodes = episodes or env['reward_estimation_episodes']
    episode_rewards = np.zeros(n_episodes)

    for i, _ in enumerate(episode_rewards):
        episodes_run += 1

        state = gym_env.reset()
        reward = 0
        done = False
        
        while not done:
            _, action = qtree.predict(state)
            next_state, reward, done, _ = gym_env.step(action)
            episode_rewards[i] += reward
            state = next_state
    
    return np.mean(episode_rewards)

def run_CUT(qtree, env, verbose=False):
    best_reward = -99999
    reward_history = []
    no_split = 0

    for i in range(env['nodes_to_grow']):
        print(f"\n==> Iteration {i}, tree size {qtree.get_size()}:")

        # Data collecting phase
        qtree = collect_data(qtree, env)
        qtree = update_value(qtree)

        if verbose:
            qtree.print_tree()
        
        # Split phase
        qtree = update_datapoints(qtree, env)
        leaf, split, score = select_split(qtree, qtree, env, verbose=False)
        if (env['splitting_criterion'] == 'ks' and score[1] < 0.05) or \
            (env['splitting_criterion'] == 'variance') or \
            (env['splitting_criterion'] == 'random'):
            print(f">> Split ({env['attributes'][split[0]][0]}, {split[1]}) is good enough! Score: {score}")
            qtree = grow_tree(qtree, leaf, None, split)
        else:
            no_split += 1
            if no_split == 3:
                print("> Three iterations without splits.")
                break

        # Upkeep phase 
        if env['qlearning_episodes'] and not env['should_qlearn_inplace']:
            print("\n> Running Q-Learning...")
            if not env['should_store_history']:
                qtree.reset_history()
            qtree = run_qlearning(qtree, env)
            qtree = update_value(qtree)

        average_reward = get_average_reward(qtree, env)
        reward_history.append((qtree.get_size(), average_reward))
        print(f"Average reward for the tree is: {average_reward}")
        if average_reward > best_reward:
            best_reward = average_reward
            best_tree = copy.deepcopy(qtree)

        # qtree.reset_history()

    print(f"Best tree, with average reward {best_reward} and size {best_tree.get_size()}:")
    if verbose:
        best_tree.print_tree()
    return best_tree, reward_history

def prune_tree(qtree, node, env):
    history = []

    if node.__class__.__name__ == "QLeaf":
        return qtree, []

    if node.left.__class__.__name__ == "QNode":
        qtree, new_history = prune_tree(qtree, node.left, env)
        history += new_history

    if node.right.__class__.__name__ == "QNode":
        qtree, new_history = prune_tree(qtree, node.right, env)
        history += new_history
    
    parent_of_leaves = (node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QLeaf")
    parent_of_subtree = (node.left.__class__.__name__ == "QNode" and node.right.__class__.__name__ == "QLeaf") or \
        (node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QNode")

    if parent_of_leaves or parent_of_subtree:
        average_reward = get_average_reward(qtree, env)
        print(f"\nThe average reward of the tree is {average_reward}.")
        changing_left_node = (node.parent and node == node.parent.left)
        
        if parent_of_leaves:
            merged_q_values = [node.left.q_values, node.right.q_values][np.argmax([np.max(node.left.q_values), np.max(node.right.q_values)])]
            merged_leaf = QLeaf(node.parent, node.parent is not None and node == node.parent.left, env['actions'], merged_q_values)

            if node.parent:
                if node == node.parent.left:
                    node.parent.left = merged_leaf
                else:
                    node.parent.right = merged_leaf
            else:
                qtree = merged_leaf

            print(f"Merged leaves of node '{env['attributes'][node.attribute][0]} <= {node.value}'!")
        
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
            print(f"Routed from node '{env['attributes'][node.attribute][0]} <= {node.value}' to its subtree '{env['attributes'][child_node.attribute][0]} <= {child_node.value}'!")

        new_average_reward = get_average_reward(qtree, env)
        print(f"Got average reward {new_average_reward} after merge.")

        if new_average_reward < (0.95 if average_reward > 0 else 1.05) * average_reward:
            print("Average reward was reduced too much. Undoing merge...")
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
            print(f"Appending {(qtree.get_size(), new_average_reward)}")
            history.append((qtree.get_size(), new_average_reward))

    return qtree, history

def run_pruned_CUT(env):
    qtree = QLeaf(parent=None, actions=env['actions'])
    history = []

    for i in range(env['cycle_length']):
        qtree, reward_history = run_CUT(qtree, env)
        history.append(reward_history)

        reward_history = []
        new_history = []
        k = 0
        while new_history or not reward_history:
            qtree, new_history = prune_tree(qtree, qtree, env)
            qtree.print_tree()
            reward_history += new_history

            k += 1
            if k > 5:
                break
        history.append(reward_history)

        if reward_history and reward_history[-1][1] > env['episode_max_score']:
            break
    
    return qtree, history

env = {
    "name": "MountainCar-v0",
    "can_render": True,
    "episode_max_score": 195,
    "should_force_episode_termination_score": True,
    "episode_termination_score": 0,
    "n_actions": 2,
    "actions": ["left", "right"],
    "n_attributes": 2,              
    "attributes": [("Car Position", "continuous", -1, -1),
                    ("Car Velocity", "continuous", -1, -1)],

    "learning_rate": 0.05,
    "discount_factor": 0.95,
    "continuous_quantiles": 10,
    "splitting_criterion": 'variance',

    "cycle_length": 50,
    "nodes_to_grow": 10, 
    "collection_episodes": 10,
    "reward_estimation_episodes": 10,
    "qlearning_episodes": 10,

    "should_store_history": True,
    "history_storage_length": 1000,
    "should_qlearn_inplace": False,
}

summary_reward = []
summary_episodes_run = []
trees = []

for _ in range(5):
    episodes_run = 0
    qtree, history = run_pruned_CUT(env)
    trees.append(copy.deepcopy(qtree))
    summary_episodes_run.append(episodes_run)
    summary_reward.append(get_average_reward(qtree, env, 100000))

for tree, episodes, reward in zip(trees, summary_episodes_run, summary_reward):
    print("\n")
    tree.print_tree()
    print(f"Reward: {reward}")
    print(f"Episodes run: {episodes}")

print(f"Average of episode rewards: {np.mean(summary_reward)}")
print(f"Average of episodes run: {np.mean(summary_episodes_run)}")
print(f"Summary reward: {summary_reward}")
print(f"Summary episodes run: {summary_episodes_run}")

# qtree, history = run_pruned_CUT(env)
# qtree.print_tree()
# save_tree(qtree)

# current_x = 0
# xticks = [[0], [1]]
# for (i, phase) in enumerate(history):
#     if len(phase) == 0:
#         continue
#     color = "blue"
#     if i % 2 == 0:
#         color = "red"

#     plt.plot(range(current_x, current_x + len(phase)), [b for (a, b) in phase], color=color)
#     xticks[0].append(current_x + len(phase) - 1)
#     xticks[1].append(phase[len(phase) - 1][0])
#     current_x += len(phase)
# xticks[0].append(current_x)
# plt.xticks(xticks[0], xticks[1])
# plt.title("Performance history")
# plt.xlabel("Tree size")
# plt.ylabel("Average reward")
# plt.show()
# print(f"Episodes run: {episodes_run}")

# view_tree_in_action(qtree, env['name'], 5, env['can_render'], verbose=True)