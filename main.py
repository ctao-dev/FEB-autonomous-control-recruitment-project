import numpy as np
from simulator import Simulator, centerline
import scipy

sim = Simulator()
horizon = 1
dt = 0.1

def pred_future(curr_x, curr_ua, curr_us):
    x_pos, y_pos, phi, v, theta = curr_x
    a, d_theta = curr_ua, curr_us
    wheel_base = 1.58

    x1 = x_pos + v * np.cos(phi) * dt
    y1 = y_pos + v * np.sin(phi) * dt
    v1 = v + a * dt
    phi1 = phi + (v * np.tan(theta)/wheel_base) * dt
    theta1 = theta + d_theta * dt

    theta1 = np.clip(theta1, -0.7, 0.7)

    return [x1, y1, phi1, v1, theta1]

def find_closest_distance(x_pos, y_pos, previous_center_dist):
    search_dist = 5.0
    search_range = np.linspace(previous_center_dist, previous_center_dist + search_dist, 10)
    search_points = centerline(search_range)

    distances = np.linalg.norm(search_points - np.array([x_pos, y_pos]), axis = 1)
    closest_dist = search_range[np.argmin(distances)]

    return max(previous_center_dist, closest_dist)

def generate_horizon(previous_center_dist):
    new_pos = 7

    arr = np.linspace(previous_center_dist + 0.01, previous_center_dist + new_pos * horizon, int(horizon/dt))

    return centerline(arr)

def lateral_acceleration(v, theta, a, d_theta, L=1.58):
    return (v ** 2 * np.tan(theta)) / L

# should add a line to cost saying if slippage then cost -> way up
def cost_function(u, curr_x, desired_x):
    cost = 0.0
    tx = curr_x

    for i in range(int(horizon/dt)):
        nx = pred_future(tx, u[i], u[i + int(horizon/dt)])
        cost += ((nx[0] - desired_x[i][0]) ** 2 + (nx[1] - desired_x[i][1]) ** 2)
        cost += 1000000 * max((u[i] ** 2 + lateral_acceleration(nx[3], nx[4], u[i], u[i + int(horizon/dt)]) ** 2 - 144), 0)
        tx = nx

    return cost

# should technically be horizon * 1/dt for each timestamp but this works for now
previous_acceleration = [4] * int(horizon/dt)
previous_steering = [0] * int(horizon/dt)
acceleration_bounds = [(-10, 4) for _ in range(int(horizon/dt))]
steering_bounds = [(-1, 1) for _ in range(int(horizon/dt))]
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

    
call = 0
control_hold = 2
tick = 0
car_RMSD = 0
l_time = 0
lap_num = 0
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


    global previous_acceleration, previous_steering, previous_center_dist, control_hold, tick
    step = tick%control_hold

    if(step == 0):
        res = optimizer(xpos, ypos, phi, v, theta)
        optimal_acceleration = res.x[:int(horizon/dt)]
        optimal_steering = res.x[int(horizon/dt):]
        previous_acceleration = optimal_acceleration
        previous_steering = optimal_steering

    previous_center_dist = find_closest_distance(xpos, ypos, previous_center_dist)
    if(previous_center_dist > 105 and lap_num == 0):
        l_time = tick/100
        lap_num += 1
    previous_center_dist = previous_center_dist%105.
    curr_center_point = centerline(previous_center_dist)
    RMSD += np.norm(curr_center_point-[xpos, ypos]);


    tick += 1

    print(tick/100, " ", previous_acceleration[0])
    return np.array([previous_acceleration[0], previous_steering[0]])




sim.set_controller(controller)
sim.run(tf=30)
sim.animate()
sim.plot()
print(np.sqrt(1/tick * car_RMSD));