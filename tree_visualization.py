import gym
import copy
import pdb
import pickle
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
import scipy.stats as stats
from qtree import QNode, QLeaf, grow_tree
from rich import print

NOP = 0
LEFT_ENGINE = 1
MAIN_ENGINE = 2
RIGHT_ENGINE = 3

def view_tree_in_action(qtree, envname, episodes=5, render=True, verbose=True):
    qtree.print_tree()
    total_rewards = []
    gym_env = gym.make(envname)

    for _ in range(episodes):
        state = gym_env.reset()
        done = False
        reward = 0
        total_reward = 0
        t = 0

        while not done:
            t += 1
            if render:
                gym_env.render()

            _, action = qtree.predict(state)

            state, reward, done, _ = gym_env.step(action)
            total_reward += reward

            if done:
                if verbose:
                    print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
                total_rewards.append(total_reward)
                break
    gym_env.close()

    print("Average reward per episode:", np.mean(total_rewards))
    return np.mean(total_rewards), np.std(total_rewards)

def run_lunarlander_optimal(qtree, envname, episodes=5, render=True, verbose=True):
    qtree.print_tree()
    total_rewards = []
    gym_env = gym.make(envname)

    for _ in range(episodes):
        state = gym_env.reset()
        done = False
        reward = 0
        total_reward = 0
        t = 0

        while not done:
            t += 1
            if render:
                gym_env.render()

            x = state
            if x[6] <= 0.5:
                if x[4] <= -0.07604862377047539:
                    if x[2] <= -0.013069174252450466:
                        action = MAIN_ENGINE
                    else:
                        action = LEFT_ENGINE
                else:
                    if x[2] <= -0.09865037351846695:
                        if x[5] <= -0.00998950470238924:
                            if x[3] <= -0.5160131752490997:
                                action = MAIN_ENGINE
                            else:
                                action = RIGHT_ENGINE
                        else:
                            action = RIGHT_ENGINE
                    else:
                        if x[1] <= 1.0102267265319824:
                            if x[3] <= -0.10799521207809448:
                                action = MAIN_ENGINE
                            else:
                                action = RIGHT_ENGINE
                        else:
                            if x[5] <= 0.052217885851860046:
                                action = LEFT_ENGINE
                            else:
                                action = RIGHT_ENGINE
            else:
                action = NOP

            state, reward, done, _ = gym_env.step(action)
            total_reward += reward

            if done:
                if verbose:
                    print("Episode finished after {} timesteps, with total reward {}".format(t+1, total_reward))
                total_rewards.append(total_reward)
                break
    gym_env.close()

    print("Average reward per episode:", np.mean(total_rewards))
    return np.mean(total_rewards), np.std(total_rewards)

def run_blackjack_optimal(episodes=100000):
    gym_env = gym.make("Blackjack-v0")
    n_episodes = episodes
    episode_rewards = np.zeros(n_episodes)

    for i, _ in enumerate(episode_rewards):
        state = gym_env.reset()
        reward = 0
        done = False
        
        while not done:
            player_sum = state[0]
            dealer_sum = state[1]
            usable_ace = state[2]

            if player_sum <= 13:
                action = 1
            else:
                action = 0

            if usable_ace:
                if dealer_sum <= 8:
                    if player_sum <= 17:
                        action = 1
                    else:
                        action = 0
                else:
                    if player_sum <= 18:
                        action = 1
                    else:
                        action = 0
            else:
                if player_sum <= 16:
                    if dealer_sum <= 6:
                        if dealer_sum <= 3:
                            if player_sum <= 12:
                                action = 1
                            else:
                                action = 0
                        else:
                            if player_sum <= 11:
                                action = 1
                            else:
                                action =0
                    else:
                        action = 1
                else:
                    action = 0
            
            # print(f"Player: {state[0]} \t Dealer: {state[1]} \t Ace: {state[2]} \t '{['STICK', 'HIT'][action]}'")
            next_state, reward, done, _ = gym_env.step(action)
            # if done:
            #     print(f"\t \t \t \t  \t \t R: {reward}")
            episode_rewards[i] += reward
            state = next_state
    
    print(f"score: {np.mean(episode_rewards)}")

if __name__ == "__main__":
    filename = "data/tree 2022-01-31 14-24_lunarlander_optimal"
    envname = "CartPole-v1"

    qtree = None
    with open(filename, 'rb') as file:
        qtree = pickle.load(file)
        file.close()
    
    view_tree_in_action(qtree, envname, episodes=50)

    # mean, std = view_tree_in_action(qtree, envname, episodes=100, render=False, verbose=True)
    # print(f"Average reward: {'{:.3f}'.format(mean)} +- {'{:.3f}'.format(std)}")

    # run_blackjack_optimal()