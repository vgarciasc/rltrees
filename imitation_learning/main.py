import ann

import gym
import numpy as np
from sklearn import tree

def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, done, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y

def label_dataset_with_model(config, model, X):
    y = model.batch_predict(X)
    y = [np.argmax(q) for q in y]
    return y

if __name__ == "__main__":
    config = {
        "name": "CartPole-v1",
        "can_render": True,
        "episode_max_score": 195,
        "should_force_episode_termination_score": True,
        "episode_termination_score": 0,
        "n_actions": 2,
        "actions": ["left", "right"],
        "n_attributes": 4,              
        "attributes": [
            ("Cart Position", "continuous", -1, -1),
            ("Cart Velocity", "continuous", -1, -1),
            ("Pole Angle", "continuous", -1, -1),
            ("Pole Angular Velocity", "continuous", -1, -1)],
    }

    # Get expert from another file
    expert = ann.MLPAgent(config, exploration_rate=0)
    expert.load("data/cartpole_nn_expert")

    # Initialize dataset and policy
    X, y = get_dataset_from_model(config, expert, episodes=100)
    dt = tree.DecisionTreeClassifier(ccp_alpha=0.01)
    dt.fit(X, y)

    best_reward = -np.inf
    best_model = None

    for i in range(50):
        # Get D'
        X2, _ = get_dataset_from_model(config, dt, 100)
        y2 = label_dataset_with_model(config, expert, X2)

        # Aggregate datasets
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))

        # Update student policy pi_i
        dt = tree.DecisionTreeClassifier(ccp_alpha=0.01)
        dt.fit(X, y)

        # Evaluate average reward from student policy pi_i
        avg_reward = get_average_reward(config, dt) # Function not pictured here
        
        # Keep tabs on best policy seen so far
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = dt