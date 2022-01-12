import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
from qtree import QNode, QLeaf, save_tree, grow_tree
from rich import print
import random

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.5
N_QUANTILES = 5

N_ACTIONS = 2
N_ATTRIBS = 3
ATTRIBUTES = [("Player's Sum", "discrete", 0, 22),
              ("Dealer's Card", "discrete", 1, 11),
              ("Usable Ace", "binary", -1, -1)]

episodes_run = 0

def collect_data(qtree, n_episodes):
    env = gym.make("Blackjack-v1")
    global episodes_run

    for T in range(1, n_episodes):
        episodes_run += 1

        state = env.reset()
        action = 0
        reward = 0
        leaf = None
        done = False
        
        while not done:
            if leaf is None:
                leaf, action = qtree.predict(state)

            if np.random.random() < 0.2:
            # if np.random.random() < max([0.2, ((0.1 / 2) * n_episodes / T)]):
                action = np.random.randint(0, N_ACTIONS)
            
            next_state, reward, done, _ = env.step(action)
            next_leaf, next_action = qtree.predict(next_state)  

            if not done:
                delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(next_leaf.q_values) - leaf.q_values[action])
            else:
            # 	reward = 0
                delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])

            # if qtree.get_size() > 1:
            #     if done:
            #         print("")
            #     print(f"-> PLAYER: {state[0]}, DEALER: {state[1]}, USABLE_ACE?: {state[2]}. ACTION: {['stick', 'hit'][action]}")
            #     print(f"-> PLAYER: {next_state[0]}, DEALER: {next_state[1]}, USABLE_ACE?: {next_state[2]}.")
            #     print(F"-> REWARD: {reward}")
            #     print(f"-> DELTA_Q: {delta_q}")
            #     qtree.print_tree()

            leaf.q_values[action] += delta_q

            leaf.q_history[action].append((state, action, next_leaf.value, reward))
            leaf.state_history.append(state)
            if len(leaf.q_history[action]) > 1000:
                leaf.q_history[action].pop(0)
                leaf.state_history.pop(0)

            leaf = next_leaf
            state = next_state
            action = next_action
    
    return qtree

def update_datapoints(node):
    if node.__class__.__name__ == "QLeaf":
        leaf = node
        
        for action_id in range(N_ACTIONS):
            for (s, a, v, r) in leaf.q_history[action_id]:
                q = r + DISCOUNT_FACTOR * v
                leaf.full_q_history[a].append((s, a, q))
    else:
        if node.left is not None:
            update_datapoints(node.left)
        if node.right is not None:
            update_datapoints(node.right)

    return node

def select_split(qtree, node, verbose=False):
    is_better_than = lambda score1, score2 : (score2[1] - score1[1] > 0.01) or (np.abs(score1[1] - score2[1]) < 0.01 and score1[0] > score2[0])

    if node.__class__.__name__ == "QLeaf":
        leaf = node
        if verbose:
            print(f"\n{'Left' if leaf.is_left else 'Right'} leaf{(' of x[' + str(leaf.parent.attribute) + '] <= ' + str(leaf.parent.value)) if leaf.parent is not None else ''}:")

        best_split = None
        best_score = [0, 1]

        for attribute_idx in range(N_ATTRIBS):
            attr_name, attr_type, start_value, end_value = ATTRIBUTES[attribute_idx]
            
            cutoffs = []
            if attr_type == "continuous":
                if len(leaf.state_history) < 100:
                    continue
                cutoffs = np.quantile([s[attribute_idx] for s in leaf.state_history], np.linspace(0.1, 0.9, N_QUANTILES))
            elif attr_type == "discrete":
                cutoffs = range(start_value, end_value)
            elif attr_type == "categorical":
                cutoffs = start_value
            elif attr_type == "binary":
                cutoffs = [0, 1]
            
            for cutoff in cutoffs:
                score = [0, 1]

                for action in range(N_ACTIONS):
                    L_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] <= cutoff]
                    R_partition = [q for (s, a, q) in leaf.full_q_history[action] if s[attribute_idx] > cutoff]

                    if L_partition and R_partition:
                        kstest = stats.ks_2samp(L_partition, R_partition)
                        score[0] += kstest[0]
                        score[1] *= kstest[1]
                        
                if verbose:
                    print(f"> Split {(attr_name, cutoff)} has score (D: {score[0]}, p-value: {score[1]})")

                if is_better_than(score, best_score):
                    best_split = (attribute_idx, cutoff)
                    best_score = score
            
        return leaf, best_split, best_score
    else:
        if node.left is not None:
            left_leaf, left_best_split, left_best_score = select_split(qtree, node.left)
        if node.right is not None:
            right_leaf, right_best_split, right_best_score = select_split(qtree, node.right)

        if is_better_than(left_best_score, right_best_score):
            return left_leaf, left_best_split, left_best_score
        else:
            return right_leaf, right_best_split, right_best_score

def run_qlearning(qtree, n_episodes):
    env = gym.make("Blackjack-v1")

    for T in range(1, n_episodes):
        global episodes_run; episodes_run += 1

        state = env.reset()
        action = None
        leaf = None
        reward = 0
        done = False
        
        while not done:
            if leaf is None:
                leaf, action = qtree.predict(state)

            if np.random.random() < 0.1:
                action = np.random.randint(0, N_ACTIONS)

            next_state, reward, done, _ = env.step(action)
            next_leaf, next_action = qtree.predict(next_state)
            
            if not done:
                delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(next_leaf.q_values) - leaf.q_values[action])
            else:
                # reward = 0
                delta_q = LEARNING_RATE * (reward + DISCOUNT_FACTOR * 0 - leaf.q_values[action])

            leaf.q_values[action] += delta_q
            # leaf.q_history[action].append(delta_q)

            state = next_state
            leaf = next_leaf
            action = next_action
    
    return qtree

def run_monte_carlo_control(qtree, n_episodes=1000):
    env = gym.make("Blackjack-v1")

    for T in range(1, n_episodes):
        global episodes_run; episodes_run += 1

        episode = []
        state = env.reset()
        done = False

        while not done:
            leaf, action = qtree.predict(state)

            if np.random.random() < 0.5 * n_episodes / T:
                action = np.random.randint(0, N_ACTIONS)
            
            next_state, reward, done, _ = env.step(action)
            # if done:
            # 	reward = 0
            episode.append((leaf, action, reward))
            state = next_state

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            leaf, action, reward = episode[t]
            
            G = DISCOUNT_FACTOR * G + reward

            has_appeared = False
            for i in range(0, t):
                leaf2, action2, _ = episode[i]
                if leaf == leaf2 and action == action2:
                    has_appeared = True
                    break
            
            if not has_appeared:
                leaf.q_history[action].append(G)
                leaf.q_values[action] = np.mean(leaf.q_history[action])
        
        # qtree.reset_history()
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

def get_average_reward(qtree, n_episodes):
    global episodes_run

    env = gym.make("Blackjack-v1")
    episode_rewards = np.zeros(n_episodes)

    for T in range(1, n_episodes):
        episodes_run += 1

        state = env.reset()
        reward = 0
        done = False
        t = 0
        
        while not done:
            t += 1
            _, action = qtree.predict(state)
            next_state, reward, done, _ = env.step(action)

            episode_rewards[T] += reward
            state = next_state
    
    return np.mean(episode_rewards)

def run_CUT(qtree, cut_iters=100, collect_data_iters=1000, q_learning_iters=5000, average_reward_iters=100, verbose=False):
    best_reward = -99999
    reward_history = []
    no_split = 0

    for i in range(cut_iters):
        print(f"\n==> Iteration {i}, tree size {qtree.get_size()}:")
        # Data collecting phase
        qtree = collect_data(qtree, collect_data_iters)
        # qtree = update_value(qtree)

        if verbose:
            qtree.print_tree()
        
        # Split phase
        qtree = update_datapoints(qtree)
        leaf, split, score = select_split(qtree, qtree, verbose=False)
        if score[1] < 0.05:
            print(f">> Split ({ATTRIBUTES[split[0]][0]}, {split[1]}) is good enough! Score: (D: {score[0]}, p: {score[1]})")
            qtree = grow_tree(qtree, leaf, None, split)
        else:
            no_split += 1
            if no_split == 3:
                print("> Three iterations without splits.")
                break

        # Upkeep phase 
        # print("\n> Running Q-Learning...")
        # qtree.reset_history()
        # qtree = run_monte_carlo_control(qtree, 20000)
        # qtree = run_qlearning(qtree, q_learning_iters)
        qtree = update_value(qtree)

        average_reward = get_average_reward(qtree, average_reward_iters)
        reward_history.append((qtree.get_size(), average_reward))
        print(f"Average reward for the tree is: {average_reward}")
        if average_reward > (1.05 if best_reward > 0 else 0.95) * best_reward:
            best_reward = average_reward
            best_tree = copy.deepcopy(qtree)

        # qtree.reset_history()

    print(f"Best tree, with average reward {best_reward} and size {best_tree.get_size()}:")
    if verbose:
        best_tree.print_tree()
    return best_tree, reward_history

def prune_tree(qtree, node, n_trials=100):
    history = []

    if node.__class__.__name__ == "QLeaf":
        return qtree, []
    
    if node.left.__class__.__name__ == "QNode":
        qtree, new_history = prune_tree(qtree, node.left, n_trials)
        history += new_history
    if node.right.__class__.__name__ == "QNode":
        qtree, new_history = prune_tree(qtree, node.right, n_trials)
        history += new_history
    
    if node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QLeaf":
        print("")
        average_reward = get_average_reward(qtree, n_trials)
        print(f"The average reward of the tree is {average_reward}.")

        was_left = None
        merged_q_values = [node.left.q_values, node.right.q_values][np.argmax([np.max(node.left.q_values), np.max(node.right.q_values)])]
        merged_leaf = QLeaf(node.parent, node.parent is not None and node == node.parent.left, node.left.actions, merged_q_values)
        if node.parent is not None:
            if node == node.parent.left:
                was_left = True
                node.parent.left = merged_leaf
            else:
                was_left = False
                node.parent.right = merged_leaf
        else:
            qtree = merged_leaf
        print(f"Merged leaves of node '{ATTRIBUTES[node.attribute][0]} <= {node.value}'!")
        
        new_average_reward = get_average_reward(qtree, n_trials)
        print(f"Got average reward {new_average_reward} after merge.")

        if new_average_reward < (0.9 if average_reward > 0 else 1.1) * average_reward:
            print("Average reward was reduced too much. Undoing merge...")
            if node.parent is not None:
                if was_left == True:
                    node.parent.left = node
                else:
                    node.parent.right = node
            else:
                qtree = node
        else:
            print(f"Appending {(qtree.get_size(), new_average_reward)}")
            history.append((qtree.get_size(), new_average_reward))

    if (node.left.__class__.__name__ == "QNode" and node.right.__class__.__name__ == "QLeaf") or \
        (node.left.__class__.__name__ == "QLeaf" and node.right.__class__.__name__ == "QNode"):
        print("")
        average_reward = get_average_reward(qtree, n_trials)
        print(f"The average reward of the tree is {average_reward}.")

        was_left = None
        if node.left.__class__.__name__ == "QNode":
            node.left.parent = node.parent
        else:
            node.right.parent = node.parent

        if node.parent is not None:
            if node == node.parent.left:
                was_left = True
                node.parent.left = node.left if node.left.__class__.__name__ == "QNode" else node.right
            else:
                was_left = False
                node.parent.right = node.left if node.left.__class__.__name__ == "QNode" else node.right
        else:
            qtree = node.left if node.left.__class__.__name__ == "QNode" else node.right
        child_node = node.left if node.left.__class__.__name__ == "QNode" else node.right
        print(f"Routed from node '{ATTRIBUTES[node.attribute][0]} <= {node.value}' to its subtree '{ATTRIBUTES[child_node.attribute][0]} <= {child_node.value}'!")

        new_average_reward = get_average_reward(qtree, n_trials)
        print(f"Got average reward {new_average_reward} after merge.")

        if new_average_reward < (0.9 if average_reward > 0 else 1.1) * average_reward:
            (node.left if node.left.__class__.__name__ == "QNode" else node.right).parent = node
            print("Average reward was r44444444117educed too much. Undoing merge...")

            if node.parent is not None:
                if was_left == True:
                    node.parent.left = node
                else:
                    node.parent.right = node
            else:
                qtree = node
        
            # new_average_reward = get_average_reward(qtree, n_trials)
            # print(f"Undid subtree routing, got average reward {new_average_reward}.")
        else:
            print(f"Appending {(qtree.get_size(), new_average_reward)}")
            history.append((qtree.get_size(), new_average_reward))

    return qtree, history

def run_pruned_CUT(overall_iters=10, cut_iters=100, collect_data_iters=1000, q_learning_iters=5000, average_reward_iters=100, pruning_reward_iters=100, episode_max=195):
    # Initializing tree
    qtree = QLeaf(parent=None, actions=["stick", "hit"])
    history = []

    for i in range(overall_iters):
        print(f"Phase {i}")
        qtree, reward_history = run_CUT(qtree, cut_iters, collect_data_iters, q_learning_iters, average_reward_iters)
        save_tree(qtree, "_growing", get_average_reward(qtree, 10))
        history.append(reward_history)

        reward_history = []
        new_history = []
        k = 0
        while len(reward_history) == 0 or len(new_history) != 0:
            qtree, new_history = prune_tree(qtree, qtree, pruning_reward_iters)
            reward_history += new_history
            save_tree(qtree, "_pruning", get_average_reward(qtree, 10))

            k += 1
            if k > 5:
                break
        history.append(reward_history)

        if len(reward_history) > 1 and reward_history[-1][1] > episode_max:
            break
    
    return qtree, history

# qtree, history = run_pruned_CUT(50, 10, 10, 10, 10, 10)
qtree, history = run_pruned_CUT(10, 20, 100, 100, 1000, 1000)
save_tree(qtree)

current_x = 0
xticks = [[0], [1]]
for (i, phase) in enumerate(history):
    if len(phase) == 0:
        continue
    color = "blue"
    if i % 2 == 0:
        color = "red"

    plt.plot(range(current_x, current_x + len(phase)), [b for (a, b) in phase], color=color)
    xticks[0].append(current_x + len(phase) - 1)
    xticks[1].append(phase[len(phase) - 1][0])
    current_x += len(phase)
xticks[0].append(current_x)
plt.xticks(xticks[0], xticks[1])
plt.title("Performance history")
plt.xlabel("Tree size")
plt.ylabel("Average reward")
plt.show()
print(f"Episodes run: {episodes_run}")

# qtree, history = run_pruned_CUT(1, 2, 20, 10, 10, 10)
# fig, axs = plt.subplots(2, 4)
# for i, attr_id in enumerate([-0.55, -0.52, -0.49, -0.46]):
# 	l_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] <= attr_id]
# 	r_partition = [q for (s, a, q) in qtree.full_q_history[0] if s[0] > attr_id]
# 	axs[0, i].hist(l_partition, bins=5, alpha=0.7, color="green", label=f"$i \leq {attr_id}$")
# 	axs[0, i].hist(r_partition, bins=5, alpha=0.7, color="red", label=f"$i > {attr_id}$")
# 	axs[0, i].legend()
# 	axs[0, i].set_title(f"$attr = {attr_id}$")
# 	axs[0, i].set_xlabel("$q(I, a)$")

# 	l_values, l_base = np.histogram(l_partition, bins=40)
# 	r_values, r_base = np.histogram(r_partition, bins=40)
# 	axs[1, i].plot(l_base[:-1], np.cumsum(l_values) / np.cumsum(l_values)[-1], linestyle='dashed', color="green", label=f"$i \leq {attr_id}$")
# 	axs[1, i].plot(r_base[:-1], np.cumsum(r_values) / np.cumsum(r_values)[-1], linestyle='dashed', color="red", label=f"$i > {attr_id}$")
# 	axs[1, i].legend()
# 	if l_partition and r_partition:
# 		axs[1, i].set_title(f"$D = {'{:.4f}'.format(stats.ks_2samp(l_partition, r_partition)[0])}, p = {'{:.4f}'.format(stats.ks_2samp(l_partition, r_partition)[1])}$")
# 	axs[1, i].set_ylim(bottom=0)
# 	axs[1, i].set_ylabel("CDF")
# 	axs[1, i].set_xlabel("$q(I, a)$")
# plt.show()
