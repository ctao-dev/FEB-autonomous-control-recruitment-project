import numpy as np
from simulator import Simulator, centerline

sim = Simulator()
TRACK_LENGTH = 105

def find_closest_point(x, y, s_range, num_points):
    """find the closest point on the centerline to the current position

    Args:
        x (float): current x position
        y (float): current y position
        s_range (tuple): (s_min, s_max) range of distances along the centerline to search
        num_points (int): number of points to sample along the centerline

    Returns:
        float: distance along the centerline of the closest point
    """
    s_values = np.linspace(s_range[0], s_range[1], num_points)
    centerline_points = centerline(s_values)
    distances = np.linalg.norm(centerline_points - np.array([x, y]), axis=1)
    closest_index = np.argmin(distances)
    return s_values[closest_index]

integral_error = 0.0
prev_error = 0.0

def compute_acceleration(current_speed, target_speed, dt, kp, ki, kd, max_accel=4, max_decel=-10):
    """determine acceleration to reach a target speed using a PID controller
    
        Args:
            current_speed (float): current speed of the car
            target_speed (float): desired speed of the car
            dt (float): time step
            kp (float): proportional gain
            ki (float): integral gain
            kd (float): derivative gain
            max_accel (float): maximum acceleration
            max_decel (float): maximum deceleration

        Returns:
            tuple: (acceleration, integral_error, error)
        """
    global integral_error, prev_error
    error = target_speed - current_speed
    integral_error += error * dt
    derivative_error = (error - prev_error) / dt
    acceleration = kp * error + ki * integral_error + kd * derivative_error
    return np.clip(acceleration, max_decel, max_accel), integral_error, error

# this is a placeholder to check if it is the first run of the controller, so we can use global search for midline
current_distance = 0.0 

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

    global integral_error, prev_error, current_distance

    # get the distance from the start of the track
    search_window = 5.0
    current_distance = find_closest_point(xpos, ypos, (current_distance, current_distance + search_window), num_points=10)%TRACK_LENGTH  # far fewer points needed for a small window104


    # determine the closest point on the centerline and a point ahead of it
    current_point = centerline(current_distance)
    ahead_point = centerline((current_distance + 1.0)%TRACK_LENGTH)  # look 1 meter ahead


    # determine the car location relative to the centerline
    tangent_vector = ahead_point - current_point
    tangent_vector /= np.linalg.norm(tangent_vector)

    to_car = np.array([xpos, ypos]) - current_point
    lateral_error = tangent_vector[0] * to_car[1] - tangent_vector[1] * to_car[0]


    # utilize Stanely controller to compute the steering angle
    k = 0.2 # gain for lateral error
    heading_error = np.arctan2(
        np.sin(np.arctan2(tangent_vector[1], tangent_vector[0]) - phi),
        np.cos(np.arctan2(tangent_vector[1], tangent_vector[0]) - phi)
    )
    steering_correction = heading_error - np.arctan2(k * lateral_error, v + 1e-5)
    desired_theta = np.clip(
    steering_correction,
    *sim.steering_limits
    )

    steering_rate = (desired_theta - theta) / 0.4

    # compute the turn acceleration and max turn speed based on the current steering angle
    acceleration_limit = 11.0

    turn_acceleration = (
        v**2 / 0.79 * np.sin(np.arctan(0.5 * np.tan(theta)))
    )

    turn_factor = abs(
        np.sin(np.arctan(0.5 * np.tan(theta)))
    )

    if turn_factor > 1e-8:
        max_turn_speed = np.sqrt(acceleration_limit * 0.79 / turn_factor)
    else:
        max_turn_speed = np.inf

    target_speed = min(6.0, max_turn_speed)

    linear_acceleration, integral_error, prev_error = compute_acceleration(
        v, target_speed, 0.01, 1.0, 0.1, 0.01
    )

    # Reserve enough acceleration budget for cornering so net acceleration remains at or below the simulator's 12 m/s^2 limit.
    max_linear_acceleration = np.sqrt(
        max(0.0, acceleration_limit**2 - turn_acceleration**2)
    )

    linear_acceleration = np.clip(
        linear_acceleration,
        -max_linear_acceleration,
        max_linear_acceleration,
    )

    #print(f"Position: ({xpos:.2f}, {ypos:.2f}), Heading: {phi:.2f}, Velocity: {v:.2f}, Steering Angle: {theta:.2f}")


    return np.array([linear_acceleration, np.clip(steering_rate, -1, 1)])




sim.set_controller(controller)
sim.run(tf=30)
sim.animate()
sim.plot()