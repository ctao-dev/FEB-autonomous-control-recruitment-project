import numpy as np
from simulator import Simulator, centerline
import scipy

sim = Simulator()

def pred_future(curr_x, curr_u, dt=0.01):
    x_pos, y_pos, phi, v, theta = curr_x
    a, d_theta = curr_u
    wheel_base = 1.58

    x1 = x_pos + v * np.cos(phi) * dt
    y1 = y_pos + v * np.sin(phi) * dt
    v1 = v + a * dt
    phi1 = phi + (v * np.tan(theta)/wheel_base) * dt
    theta1 = theta + d_theta * dt

    return [x1, y1, v1, theta1, phi1]

def find_closest_distance(x_pos, y_pos, previous_center_dist):
    search_dist = 5.0
    search_range = np.linspace(previous_center_dist, previous_center_dist + search_dist, 10)
    search_points = centerline(search_range)

    distances = np.linalg.norm(search_points - np.array(x_pos, y_pos), axis = 1)
    closest_dist = np.argmin(distances)

    return closest_dist

def generate_horizon(previous_center_dist):
    new_pos = 0.07
    curr_pos = previous_center_dist
    horizon = 3

    arr = np.zeros(horizon * 100)

    for i in range(horizon * 100):
        curr_pos += new_pos
        arr[i] = curr_pos

    return centerline(arr)


# should add a line to cost saying if slippage then cost -> way up
def cost_function(u, curr_x, desired_x, horizon):
    cost = 0.0
    K = 1.0

    for i in range(horizon * 100):
        nx = pred_future(curr_x)
        cost += ((nx[0] - desired_x[i][0]) ** 2 + (nx[1] - desired_x[i][1]) ** 2 + K * nx[3] ** 2)

    return cost

# need to program this
def optimizer(xpos, ypos, phi, v, theta, a, dtheta):
    x0 = np.array([xpos, ypos, phi, v, theta])
    u0 = np.array([a, dtheta])


    

def controller(x):
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    xpos   = x[0]                   # current x position
    ypos   = x[1]                   # current y position
    phi    = np.mod(x[2], 2*np.pi)  # current heading (radians)
    v      = x[3]                   # current velocity
    theta   = x[4]                  # current steering angle

    
    
    ... # YOUR CODE HERE

    return np.array([0,0])




sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()