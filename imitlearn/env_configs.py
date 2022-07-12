import pdb
import numpy as np

# snake configuration
import sneks
from snake.snake import extract_features

config_CP = {
    "name": "CartPole-v1",
    "can_render": True,
    "skip_first_frame": False,
    "episode_max_score": 195,
    "should_force_episode_termination_score": True,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
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

config_LL = {
    "name": "LunarLander-v2",
    "can_render": True,
    "skip_first_frame": False,
    "n_actions": 4,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
    "actions": ["nop", "left engine", "main engine", "right engine"],
    "n_attributes": 8,              
    "attributes": [
        ("X Position", "continuous", -1, -1),
        ("Y Position", "continuous", -1, -1),
        ("X Velocity", "continuous", -1, -1),
        ("Y Velocity", "continuous", -1, -1),
        ("Angle", "continuous", -1, -1),
        ("Angular Velocity", "continuous", -1, -1),
        ("Leg 1 is Touching", "binary", [0, 1], -1),
        ("Leg 2 is Touching", "binary", [0, 1], -1)],
}

config_MC = {
    "name": "MountainCar-v0",
    "can_render": True,
    "skip_first_frame": False,
    "episode_max_score": 195,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
    "n_actions": 3,
    "actions": ["left", "nop", "right"],
    "n_attributes": 2,              
    "attributes": [
        ("Car Position", "continuous", -1, -1),
        ("Car Velocity", "continuous", -1, -1)],
}

config_BJ = {
    "name": "Blackjack-v0",
    "can_render": False,
    "skip_first_frame": False,
    "episode_max_score": 1,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": None,
    "n_actions": 2,
    "actions": ["stick", "hit"],
    "n_attributes": 3,
    "attributes": [
        ("Player's Sum", "discrete", 0, 22),
        ("Dealer's Card", "discrete", 1, 11),
        ("Usable Ace", "binary", -1, -1)],
}

config_SN = {
    "name": "babysnek-raw-16-v1",
    "can_render": True,
    "render_delay_ms": 100,
    "episode_max_score": 16,
    "should_force_episode_termination_score": False,
    "skip_first_frame": True,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda env, s1, s2 : extract_features(s1, s2),
    "episode_termination_score": None,
    "n_actions": 4,
    "actions": ["forward", "left", "right"],
    # "n_attributes": 36,
    "n_attributes": 5,
    "attributes": [
        ("Distance Left", "continuous", 0, 1),
        ("Distance Right", "continuous", 0, 1),
        ("Distance Up", "continuous", 0, 1),
        ("Distance Down", "continuous", 0, 1),
        ("Angle to Apple", "continuous", -1, 1)]
}

def get_config(task_name):
    if task_name == "cartpole":
        return config_CP
    elif task_name == "mountain_car":
        return config_MC
    elif task_name == "lunar_lander":
        return config_LL
    elif task_name == "blackjack":
        return config_BJ
    elif task_name == "snake":
        return config_SN
        
    print(f"Invalid task_name {task_name}.")
    return None