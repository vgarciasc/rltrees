config_CP = {
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

config_LL = {
    "name": "LunarLander-v2",
    "can_render": True,
    "n_actions": 4,
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
    "episode_max_score": 195,
    "should_force_episode_termination_score": False,
    "episode_termination_score": 0,
    "n_actions": 3,
    "actions": ["left", "nop", "right"],
    "n_attributes": 2,              
    "attributes": [
        ("Car Position", "continuous", -1, -1),
        ("Car Velocity", "continuous", -1, -1)],
}

def get_config(task_name):
    if task_name == "cartpole":
        return config_CP
    elif task_name == "mountain_car":
        return config_MC
    elif task_name == "lunar_lander":
        return config_LL
        
    print(f"Invalid task_name {task_name}.")
    return None