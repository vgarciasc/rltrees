from distilled_tree import DistilledTree
from il import visualize_model

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

    dt = DistilledTree(config)
    dt.load_model("data/dagger_best_tree_CartPole-v1")
    
    dt.save_fig()
    visualize_model(config, dt, 10)
