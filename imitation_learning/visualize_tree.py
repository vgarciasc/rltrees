from distilled_tree import DistilledTree
from il import visualize_model
from imitation_learning.il import get_average_reward
from qtree import save_tree_from_print
from rich import print

import gym

if __name__ == "__main__":
    # config = {
    #     "name": "CartPole-v1",
    #     "can_render": True,
    #     "episode_max_score": 195,
    #     "should_force_episode_termination_score": True,
    #     "episode_termination_score": 0,
    #     "n_actions": 2,
    #     "actions": ["left", "right"],
    #     "n_attributes": 4,              
    #     "attributes": [
    #         ("Cart Position", "continuous", -1, -1),
    #         ("Cart Velocity", "continuous", -1, -1),
    #         ("Pole Angle", "continuous", -1, -1),
    #         ("Pole Angular Velocity", "continuous", -1, -1)],
    # }

    # config = {
    #     "name": "LunarLander-v2",
    #     "can_render": True,
    #     "n_actions": 4,
    #     "actions": ["nop", "left engine", "main engine", "right engine"],
    #     "n_attributes": 8,              
    #     "attributes": [
    #         ("X Position", "continuous", -1, -1),
    #         ("Y Position", "continuous", -1, -1),
    #         ("X Velocity", "continuous", -1, -1),
    #         ("Y Velocity", "continuous", -1, -1),
    #         ("Angle", "continuous", -1, -1),
    #         ("Angular Velocity", "continuous", -1, -1),
    #         ("Leg 1 is Touching", "binary", [0, 1], -1),
    #         ("Leg 2 is Touching", "binary", [0, 1], -1)],
    # }
    
    config = {
		"name": "MountainCar-v0",
        "can_render": True,
        "episode_max_score": 195,
        "should_force_episode_termination_score": False,
        "episode_termination_score": 0,
        "n_actions": 3,
        "actions": ["left", "nop", "right"],
        "n_attributes": 2,              
        "attributes": [("Car Position", "continuous", -1, -1),
                       ("Car Velocity", "continuous", -1, -1)],
    }

    filename = "data/dagger_best_tree_MountainCar-v0"

    dt = DistilledTree(config)
    dt.load_model(filename)
    
    dt.save_fig()
    visualize_model(config, dt, 10)
    get_average_reward(config, dt, 100, verbose=True)

    qtree = dt.get_as_qtree()
    save_tree_from_print(
        qtree,
        config['actions'],
        f"_dagger_best_tree_MountainCar-v0")