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
n_robots = 10
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
            self.gen_heading()
            self.heading_cooldown = 0  # timesteps until the heading can be changed again

        def gen_heading(self):
            self.heading = np.random.randint(-1, 2, size=2)
            #while np.equal(self.heading, np.array([0, 0])):
            while np.all(self.heading==np.array([0, 0])):
                self.heading = np.random.randint(-1, 2, size=2)

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
        robot1.map = np.copy(new_map)
        robot2.map = np.copy(new_map)
        return

    def sample_soil(self, robot: Robot, sample_size=1):
        # Let this robot sample the soil it's on for a specified size. Updates the robots internal map

        # Form the indexes and make sure it will be inside the map
        lower_x = robot.position[0] - sample_size
        upper_x = robot.position[0] + sample_size
        lower_y = robot.position[1] - sample_size
        upper_y = robot.position[1] + sample_size
        if lower_x < 0:
            lower_x = 0
        if lower_y < 0:
            lower_y = 0
        if upper_x > self.ground_truth.shape[1]:
            upper_x = self.ground_truth.shape[1]
        if upper_y > self.ground_truth.shape[0]:
            upper_y = self.ground_truth.shape[0]

        # Apply the local area from the actual map to the robot map
        robot.map[lower_y:upper_y, lower_x:upper_x] = self.ground_truth[lower_y:upper_y, lower_x:upper_x]
        return

    def heading_correct(self, robot: Robot):
        robot.heading = np.random.randint(-1, 2, size=2)
        robot.heading_cooldown = 3
        while np.array_equal(robot.heading, np.array([0, 0])):
            robot.heading = np.random.randint(-1, 2, size=2)
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


def plot_world(map_truth, robots: np.array = None):
    # Matplot a map of the field, showing the 3 levels for nitrogen, phosphorus and potassium
    # as well as the positions of the robots and what they've detected for each element

    # Remove this line if you want to run tests and print the output
    #clear_output(wait=True)
    t0 = time.time()
    plt.gcf()
    for i in range(0, 3):
        for j in range(0, 2):
            ax[j, i].clear()

    titles = ['Ground truth nitrogen', 'Ground truth phosphorus', 'Ground truth potassium']
    # Plot map
    for i in range(0, 3):
        ax[0, i].pcolormesh(x, y, map_truth[:, :, i])
        ax[0, i].set_title(titles[i], fontsize=5)
        ax[0, i].set_xlim([0, map_truth.shape[1]])
        ax[0, i].set_ylim([0, map_truth.shape[0]])
        plt.gca().set_aspect('equal', adjustable='box')

    titles2 = ['Detected nitrogen', 'Detected phosphorus', 'Detected potassium']
    # If a 2D numpy array was given for the robots, plot them
    if robots is not None:
        for i in range(0, 3):
            ax[1, i].pcolormesh(x, y, robots[0].map[:, :, i])

        positions = np.ones((0, 2))
        for r in robots:
            r: Swarm.Robot
            positions = np.vstack((positions, r.position))

        for i in range(0, 3):
            ax[0, i].scatter(positions[:, 0] + 0.5, positions[:, 1] + 0.5, c='w')
            ax[1, i].scatter(positions[:, 0] + 0.5, positions[:, 1] + 0.5, c='w')

            # Mark robot 0
            ax[0, i].scatter(positions[0, 0] + 0.5, positions[0, 1] + 0.5, c='k', s=0.25)
            ax[1, i].scatter(positions[0, 0] + 0.5, positions[0, 1] + 0.5, c='k', s=0.25)

            ax[1, i].set_xlim([0, map_truth.shape[1]])
            ax[1, i].set_ylim([0, map_truth.shape[0]])
            ax[1, i].set_title(titles2[i], fontsize=5)
            #plt.gca().set_aspect('equal', adjustable='box')


    plt.draw()
    print("Map update took: ", time.time()-t0)
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
    left_map = False

    # check lower
    if np.any(robot.position < 0):
        robot.position[robot.position < 0] = 0
        left_map = True

    # check x
    if (robot.position[0] > world_size_x - 1):
        robot.position[0] = world_size_x - 1
        left_map = True

    # check y
    if (robot.position[1] > world_size_y - 1):
        robot.position[1] = world_size_y - 1
        left_map = True

    if left_map:
        robot.heading = np.random.randint(-1, 2, size=2)
        robot.heading_cooldown = 3
        while np.array_equal(robot.heading, np.array([0, 0])):
            robot.heading = np.random.randint(-1, 2, size=2)
    return





plt.ion()
plt.show()
time.sleep(0.5)
fig, ax = plt.subplots(2, 3, figsize=(5, 5))
fig.tight_layout()


map_truth = generate_map(world_size_x, world_size_y, n_points, blur_size)
swarm = Swarm(n_robots, map_truth)
plot_world(map_truth, swarm.swarm_list)
time.sleep(3)

# Test, robot 0 the actual map
# swarm.swarm_list[0].map = map_truth

while True:
    # Motion model
    for r in swarm.swarm_list:
        # Reduce cooldown timer if above 0
        if r.heading_cooldown > 0:
            r.heading_cooldown -= 1

        # check proximity
        if swarm.get_within_d(r, 5).shape[0] > 0:
            swarm.heading_correct(r)
        # if True:
            # random_move = np.random.randint(-2, 3, size=2)
            # r.position += random_move

        # Move the robot
        r.position += r.heading
        verify_single_robot_position(r, world_size_x, world_size_y)

        # Sample at robots new position
        swarm.sample_soil(r, 5)

    # Measurement model
    positions = np.ones((0, 2))
    for r in swarm.swarm_list:
        r: Swarm.Robot
        positions = np.vstack((positions, r.position))


    # Measurement Sharing
    for r in swarm.swarm_list:
        in_range = swarm.get_within_d(r, 15)
        for r2 in in_range:
            swarm.share_map(r, r2)

    # for r in swarm.swarm_list:
    #    print(np.equal(r.map, map_truth))

    # TODO exploration algorithm
    plot_world(map_truth, swarm.swarm_list)
    time.sleep(0.25)
