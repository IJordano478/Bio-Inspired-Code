# TEST#
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, ion
#from IPython.display import clear_output
from scipy.ndimage import gaussian_filter
#% matplotlibinline

# Seed the sim
random.seed(1)
np.random.seed(1)

# Program Params
n_points = random.randint(0, 100)
world_size_x = 50
world_size_y = 100
blur_size = 5
n_robots = 20
swarm = np.zeros((n_robots, 2))

# Uncomment to start top right
# swarm[:, 0] = np.random.randint(world_size_x-1, world_size_x, size=(n_robots))
# swarm[:, 1] = np.random.randint(world_size_y-1, world_size_y, size=(n_robots))

# Initial declarations
x = np.arange(0, world_size_x + 1, 1)
y = np.arange(0, world_size_y + 1, 1)
Z = np.ones((y.shape[0] - 1, x.shape[0] - 1)) * -1  # Intensity


class Swarm:

    def __init__(self, n_robots: int, map: np.array):
        self.swarm_list = np.ndarray(shape=n_robots, dtype=self.Robot, order='F')
        self.ground_truth = map

        for i in range(0, n_robots):
            start_pos = np.array([random.randint(0, map.shape[1]), random.randint(0, map.shape[0])])
            self.swarm_list[i] = self.Robot(start_pos, map.shape)

    class Robot:
        def __init__(self, coord, map_shape):
            self.position = coord
            self.map = np.zeros(map_shape)

    def get_closest_n(self, robot, n=1):
        # Generate a 2d array of all robot positions
        robot: Swarm.Robot
        positions = np.ones((0, 2))
        for r in self.swarm_list:
            r: Swarm.Robot
            positions = np.vstack((positions, r.position))

        # Euclidean distance and return the objects of closest N robots
        d = robot.position - positions
        indexlist = np.argsort(np.linalg.norm(d, axis=1))
        return self.swarm_list[indexlist[1:n+1]]

    def get_within_d(self, robot, max_d=10):
        # Generate a 2d array of all robot positions
        robot: Swarm.Robot
        positions = np.ones((0, 2))
        for r in self.swarm_list:
            r: Swarm.Robot
            positions = np.vstack((positions, r.position))

        # Euclidean distance and return the objects within specified d
        d = np.linalg.norm(robot.position - positions, axis=1)
        in_range_robots = self.swarm_list[d < max_d]
        return in_range_robots[1:]

    def share_map(self, robot1, robot2):
        robot1: Swarm.Robot
        robot2: Swarm.Robot
        new_map = np.maximum(robot1.map, robot2.map)
        robot1.map = new_map
        robot2.map = new_map
        return

def generate_map(world_size_x, world_size_y, n_points, blur_size):
    # Generate the number of points, place on the map, and then just gaussian blur the points to make a distribution
    # Returns: map_truth,  a map of the world containing randomised nitrogen, phosphorus and potassium values

    map_truth = np.zeros((world_size_y, world_size_x, 3))
    for i in range(0, 3):

        x = np.random.randint(0, world_size_x, size=n_points)
        y = np.random.randint(0, world_size_y, size=n_points)
        z = np.zeros((world_size_y, world_size_x))

        # Place generated points on map and then apply blurring gaussian kernel
        # NB: we may want to take intensity for accuracy calculations later as it gives the center of gaussians
        for ix, iy in np.vstack((x, y)).transpose():
            z[iy, ix] = 1.
        intensity = gaussian_filter(z, sigma=blur_size)

        map_truth[:, :, i] = intensity
    return map_truth


def plot_world(map_truth, robots=None):
    # Matplot a map of the field, showing the 3 levels for nitrogen, phosphorus and potassium
    # as well as the positions of the robots and what they've detected for each element

    # Remove this line if you want to run tests and print the output
    #clear_output(wait=True)
    plt.gcf()
    for i in range(0, 3):
        for j in range(0, 2):
            ax[j, i].clear()

    titles = ['Ground truth nitrogen', 'Ground truth phosphorus', 'Ground truth potassium']
    # Plot map

    for i in range(0, 3):
        ax[0, i].pcolormesh(x, y, map_truth[:, :, i])
        ax[0, i].title.set_text(titles[i])
        plt.gca().set_aspect('equal', adjustable='box')

    titles2 = ['Detected nitrogen', 'Detected phosphorus', 'Detected potassium']
    # If a 2D numpy array was given for the robots, plot them
    if robots is not None:
        for i in range(0, 3):
            ax[0, i].scatter(robots[:, 0] + 0.5, robots[:, 1] + 0.5, c='w')
            ax[1, i].scatter(robots[:, 0] + 0.5, robots[:, 1] + 0.5, c='k')
            ax[1, i].set_xlim([0, map_truth.shape[1]])
            ax[1, i].set_ylim([0, map_truth.shape[0]])
            ax[1, i].title.set_text(titles2[i])
            plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.001)



def verify_robot_position(swarm, world_size_x, world_size_y):
    # check lower
    swarm[swarm < 0] = 0

    # check x
    swarm[swarm[:, 0] > world_size_x - 1, 0] = world_size_x - 1

    # check y
    swarm[swarm[:, 1] > world_size_y - 1, 1] = world_size_y - 1
    return swarm


def verify_single_robot_position(robot: Swarm.Robot, world_size_x, world_size_y):
    # check lower
    robot.position[robot.position < 0] = 0

    # check x
    if (robot.position[0] > world_size_x - 1):
        robot.position[0] = world_size_x - 1

    # check y
    if (robot.position[1] > world_size_y - 1):
        robot.position[1] = world_size_y - 1

    return





# fig, ax = plt.subplots(figsize=(10, 10))
plt.ion()
plt.show()
time.sleep(0.5)
fig, ax = plt.subplots(2, 3, figsize=(5, 5))


map_truth = generate_map(world_size_x, world_size_y, n_points, blur_size)
swarm = Swarm(20, map_truth)
plot_world(map_truth)
time.sleep(3)

# for i in range(0, 10):
while (True):
    for r in swarm.swarm_list:
        if swarm.get_within_d(r).shape[0] > 0:
            random_move = np.random.randint(-2, 3, size=2)
            r.position += random_move
            verify_single_robot_position(r, world_size_x, world_size_y)
        else:
            pass

    positions = np.ones((0, 2))
    for r in swarm.swarm_list:
        r: Swarm.Robot
        positions = np.vstack((positions, r.position))
    plot_world(map_truth, positions)
    # random_move = np.random.randint(-2, 3, size=(n_robots, 2))
    # swarm = swarm + random_move
    # swarm = verify_robot_position(swarm, world_size_x, world_size_y)
    #
    # plot_world(map_truth, swarm)
    # print("looped")
    time.sleep(0.5)
