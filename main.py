"""
I programmed the PID and Stanley Controller with the help of an LLM to understand how controller implementation works in Python
"""
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
    candidate_integral = integral_error + error * dt
    derivative_error = (error - prev_error) / dt
    acceleration = kp * error + ki * candidate_integral + kd * derivative_error
    clipped_acceleration = np.clip(acceleration, max_decel, max_accel)

    # Do not keep integrating while the actuator is already saturated.
    if clipped_acceleration == acceleration or np.sign(error) != np.sign(acceleration - clipped_acceleration):
        integral_error = candidate_integral
    return clipped_acceleration, integral_error, error


WHEELBASE_HALF = 0.79
NET_ACCELERATION_LIMIT = 11.5
TURN_SPEED_ACCELERATION = 7.0
MAX_TARGET_SPEED = 15.0
MIN_SPEED_PREVIEW = 1.5
MAX_SPEED_PREVIEW = 6.0
SPEED_PREVIEW_GAIN = 0.5

current_distance = 0 # Initialize the current distance along the track

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
    current_distance = find_closest_point(xpos, ypos, (current_distance, current_distance + search_window), num_points=10)%TRACK_LENGTH  # far fewer points needed for a small window


    # determine the closest point on the centerline and a point ahead of it
    current_point = centerline(current_distance)
    ahead_point = centerline((current_distance + 1.0)%TRACK_LENGTH)  # look 1 meter ahead


    # determine the car location relative to the centerline
    tangent_vector = ahead_point - current_point
    tangent_vector /= np.linalg.norm(tangent_vector)

    to_car = np.array([xpos, ypos]) - current_point
    lateral_error = tangent_vector[0] * to_car[1] - tangent_vector[1] * to_car[0]


    # Utilize Stanely controller to compute the steering angle
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

    # Plan speed from both the current and requested steering angles. This
    # slows the car before the steering angle has reached a sharp turn.
    planning_theta = max(abs(theta), abs(desired_theta))
    turn_factor = abs(
        np.sin(np.arctan(0.5 * np.tan(planning_theta)))
    )
    if turn_factor > 1e-8:
        max_turn_speed = np.sqrt(
            TURN_SPEED_ACCELERATION * WHEELBASE_HALF / turn_factor
        )
    else:
        max_turn_speed = np.inf

    # Estimate curvature farther along the track so braking starts before
    # the current steering angle becomes large.
    speed_preview = np.clip(
        MIN_SPEED_PREVIEW + SPEED_PREVIEW_GAIN * max(v, 0.0),
        MIN_SPEED_PREVIEW,
        MAX_SPEED_PREVIEW,
    )
    preview_start = centerline((current_distance + speed_preview) % TRACK_LENGTH)
    preview_end = centerline(
        (current_distance + speed_preview + 1.0) % TRACK_LENGTH
    )
    preview_tangent = preview_end - preview_start
    preview_tangent /= np.linalg.norm(preview_tangent)
    current_heading = np.arctan2(tangent_vector[1], tangent_vector[0])
    preview_heading = np.arctan2(preview_tangent[1], preview_tangent[0])
    heading_change = np.arctan2(
        np.sin(preview_heading - current_heading),
        np.cos(preview_heading - current_heading),
    )
    curvature = abs(heading_change) / speed_preview
    if curvature > 1e-8:
        preview_turn_speed = np.sqrt(TURN_SPEED_ACCELERATION / curvature)
        max_turn_speed = min(max_turn_speed, preview_turn_speed)

    target_speed = min(MAX_TARGET_SPEED, max_turn_speed)

    linear_acceleration, integral_error, prev_error = compute_acceleration(
        v, target_speed, 0.01, 1.5, 0.1, 0.01
    )

    # Use the simulator's exact lateral-acceleration model and reserve the
    # remaining net-acceleration budget for longitudinal acceleration.
    turn_acceleration = (
        v**2 / WHEELBASE_HALF * np.sin(np.arctan(0.5 * np.tan(theta)))
    )
    # print(turn_acceleration) # Debugging to see if the turn acceleration is exceeding the limit
    max_forward_acceleration = np.sqrt(
        max(0.0, NET_ACCELERATION_LIMIT**2 - turn_acceleration**2)
    )
    max_braking_acceleration = np.sqrt(
        max(0.0, 12.0**2 - turn_acceleration**2)
    )

    linear_acceleration = np.clip(
        linear_acceleration,
        -max_braking_acceleration,
        max_forward_acceleration,
    )

    #print(linear_acceleration) # Debugging linear acceleration to see if it is exceeding the limit

    #print(f"Position: ({xpos:.2f}, {ypos:.2f}), Heading: {phi:.2f}, Velocity: {v:.2f}, Steering Angle: {theta:.2f}")


    return np.array([linear_acceleration, np.clip(steering_rate, -1, 1)])




sim.set_controller(controller)
sim.run(tf=30)
sim.animate()
sim.plot()
