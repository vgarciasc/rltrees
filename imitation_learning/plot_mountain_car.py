from imitation_learning.distilled_tree import DistilledTree
from imitation_learning.utils import printv, load_dataset
from matplotlib.colors import ListedColormap
from rich import print

import matplotlib.pyplot as plt
import numpy as np

import pdb

colors = ["blue", "grey", "orange"]
cmap = ListedColormap(colors)

if __name__ == "__main__":
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

    filename = "data/mountain_car_ann"

    X, y = load_dataset(filename + "_dataset")
    print(f"Dataset size: {len(y)}")

    dt = DistilledTree(config)
    dt.load_model("data/best_bc_mountaincar")

    plot_step = 0.001
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    pdb.set_trace()
    Z = dt.model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)

    for (action_id, action) in enumerate(config['actions']):
        car_pos = [x1 for (i, [x1, x2]) in enumerate(X) if y[i] == action_id]
        car_vel = [x2 for (i, [x1, x2]) in enumerate(X) if y[i] == action_id]

        plt.scatter(car_pos, car_vel, label=action, color=colors[action_id])

    plt.legend()
    plt.xlabel("Car Position")
    plt.ylabel("Car Velocity")
    plt.title("Expert decision boundaries for MountainCar-v0")
    plt.show()