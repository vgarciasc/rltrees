import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from qtree import QNode, QLeaf

N_ACTIONS = 2
ACTIONS = ["left", "right"]

def mdp_step(state, action):
	next_state, reward = state, 0
	done = False

	if (state == 0 or state == 1):
		next_state = 2
	if (state == 2):
		next_state = 1 if action == 0 else 0
	
	rewards = [[-1.0, -1.3, -0.7], [+0.7, +1.0, -0.5]]
	reward = rewards[action][state]

	return next_state, reward, done

def run_qlearning(qtree):
    state = 0
    reward = 0
    done = False

    for _ in range(10000):
        leaf, action = qtree.predict([state])

        # if np.random.random() < 0.1:
        #     action = np.random.randint(0, N_ACTIONS)
        action = np.random.randint(0, N_ACTIONS)

        next_state, reward, done = mdp_step(state, action)
        next_leaf, next_action = qtree.predict([next_state])
        
        if not done:
            delta_q = learning_rate * (reward + discount_factor * np.max(next_leaf.q_values) - leaf.q_values[action])
        else:
            delta_q = learning_rate * (reward + discount_factor * 0 - leaf.q_values[action])

        leaf.q_values[action] += delta_q
        leaf.q_history[action].append(delta_q)

        state = next_state
    return qtree

def run_monte_carlo_control(qtree, num_episodes=1000, iter_per_episode=1000):
    for T in range(num_episodes):
        episode = []
        state = 0

        for t in range(iter_per_episode):
            leaf, action = qtree.predict([state])
            # if np.random.random() < 0.1:
            if np.random.random() < (10 / (T + 1)):
                action = np.random.randint(0, N_ACTIONS)
            
            next_state, reward, _ = mdp_step(state, action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            leaf, _ = qtree.predict([state])
            
            G = discount_factor * G + reward

            has_appeared = False
            for i in range(0, t):
                state2, action2, _ = episode[i]
                if state == state2 and action == action2:
                    has_appeared = True
                    break
            
            if not has_appeared:
                leaf.q_history[action].append(G)
                leaf.q_values[action] = np.mean(leaf.q_history[action])
        
        qtree.reset_history()
    return qtree

eps = 1
learning_rate = 0.1
discount_factor = 0.9

# Qtree-1
qtree1 = QNode((0, 0))
leaf1_x0 = QLeaf(qtree1, is_left=True, actions=ACTIONS)
node1_1 = QNode((0, 1))
leaf1_x1 = QLeaf(node1_1, is_left=True, actions=ACTIONS)
leaf1_x2 = QLeaf(node1_1, is_left=False, actions=ACTIONS)
qtree1.set_left_child(leaf1_x0)
qtree1.set_right_child(node1_1)
node1_1.set_left_child(leaf1_x1)
node1_1.set_right_child(leaf1_x2)

# QTree-2
qtree2 = QNode((0, 1))
leaf2_x0x1 = QLeaf(qtree2, is_left=True, actions=ACTIONS)
leaf2_x2 = QLeaf(qtree2, is_left=False, actions=ACTIONS)
qtree2.set_left_child(leaf2_x0x1)
qtree2.set_right_child(leaf2_x2)

# qtree = run_qlearning(qtree2)
# qtree.print_tree()

qtree = run_monte_carlo_control(qtree2, 1000, 1000)

qtree.print_tree()