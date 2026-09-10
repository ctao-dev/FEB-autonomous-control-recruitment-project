import numpy as np
from simulator import Simulator, centerline
import scipy

sim = Simulator()

def pred_future(curr_x, curr_ua, curr_us, dt=0.01):
    x_pos, y_pos, phi, v, theta = curr_x
    a, d_theta = curr_ua, curr_us
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
    horizon = 1

    arr = np.zeros(horizon * 100)

    for i in range(horizon * 100):
        curr_pos += new_pos
        arr[i] = curr_pos

    return centerline(arr)


# should add a line to cost saying if slippage then cost -> way up
def cost_function(u, curr_x, desired_x, horizon=1):
    cost = 0.0
    K = 1.0

    for i in range(horizon * 100):
        nx = pred_future(curr_x, u[i], u[i + 100])
        cost += ((nx[0] - desired_x[i][0]) ** 2 + (nx[1] - desired_x[i][1]) ** 2 + K * nx[3] ** 2)

    return cost

# should technically be horizon * 100 for each timestamp but this works for now
previous_acceleration = [4] * 100
previous_steering = [0] * 100
acceleration_bounds = [(-10, 4) for _ in range(100)]
steering_bounds = [(-1, 1) for _ in range(100)]
previous_center_dist = 0
net_bounds = acceleration_bounds + steering_bounds
def optimizer(xpos, ypos, phi, v, theta):
    x0 = np.array([xpos, ypos, phi, v, theta])
    new_acceleration = np.concatenate((previous_acceleration[1:], previous_acceleration[-1:]))
    new_steering = np.concatenate((previous_steering[1:], previous_steering[-1:]))
    u0 = np.concatenate((new_acceleration, new_steering))

    desired_pos = generate_horizon(previous_center_dist)
    
    res = scipy.optimize.minimize(cost_function, u0, args=(x0, desired_pos), method = 'SLSQP', bounds=net_bounds, options=dict(maxiter=100),)
    return res

    

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

    res = optimizer(xpos, ypos, phi, v, theta)
    optimal_acceleration = res.x[:100]
    optimal_steering = res.x[100:]

    previous_acceleration = optimal_acceleration
    previous_steering = optimal_steering

    return np.array([optimal_acceleration[0], optimal_steering[0]])




sim.set_controller(controller)
sim.run(tf=20)
sim.animate(save=True)
sim.plot()