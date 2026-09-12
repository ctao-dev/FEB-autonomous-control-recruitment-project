import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms

_centerline = ca.external(
    'centerline', 
    ca.Importer(
        'assets/cline_func.c',
        'shell'
    )
)
centerline = np.vectorize(
    lambda x: _centerline(x).toarray().flatten(),
    signature='()->(2)',
    doc="""get track centerline at distance `x` from the start.

        Args:
            x (float or array-like): distance along centerline to get.

        Returns:
            array: shape (2,) if `x` is a float or (N, 2) if `x` is an array-like of length N.
        """
)


class Simulator:
    def __init__(self):
        """initializes a Simulator

        Args:
            controller_callback (function): function that takes in a numpy array of [x, y, theta, v, phi] and outputs a control [a, phidot]
        """
        self.dynamics = ca.external('F', ca.Importer('assets/system_dynamics.c', 'shell'))
        self.left_cones = np.load('assets/left.npy')
        self.right_cones = np.load('assets/right.npy')
        self._cones = np.concatenate([self.left_cones, self.right_cones], axis=0)
        self.car_outline = np.load('assets/pts_mat.npy')
        self.A = np.load('assets/a_mat.npy')
        self.b = np.load('assets/b_mat.npy')
        self._steering_limits = (-0.7, 0.7)
        self._accel_limits = (-10, 4)
        self._steering_vel_limits = (-1, 1)
    def set_controller(self, controller_callback):
        self.cb = controller_callback
    @property
    def cones(self):
        """get all cone locations.

        Returns:
            array: shape (N, 2) array of cone coordinates.
        """
        return self._cones
    @property
    def steering_limits(self):
        """get the feasible range for the front tires.

        Returns:
            (float, float): (min, max) angle of front tires, in radians.
        """
        return self._steering_limits
    @property 
    def lbu(self):
        """get the lower bound on the control matrix $u=[a, \dot{\phi}]$

        Returns:
            array: shape (2,), lower bound for [acceleration, steering velocity]
        """
        return np.array([self._accel_limits[0], self._steering_vel_limits[0]])
    @property
    def ubu(self):
        """get the upper bound on the control matrix $u=[a, \dot{\phi}]$

        Returns:
            array: shape (2,), upper bound for [acceleration, steering velocity]
        """
        return np.array([self._accel_limits[1], self._steering_vel_limits[1]])
    @property
    def car_vertices(self):
        return self.car_outline
    def R(self, theta):
        """SO(2) matrix constructor

        Args:
            theta (float): angle

        Returns:
            array: 2x2 rotation matrix for the given angle
        """
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    def _check_collision(self, state):
        return np.any(np.all(self.A@self.R(-state[2])@((self._cones - state[0:2]).T) < self.b[:, np.newaxis], axis=0), axis=0)
    def _get_accel(self, state, control):
        l = 0.79 # wheelbase=1.58, this is half
        return np.sqrt(
            control[0]**2
         + ((state[3]**2 / l) * np.sin(np.arctan(0.5 * np.tan(state[4]))))**2
        )
    def _check_accel(self, state, control):
        return self._get_accel(state, control) > 12 
    def run(self, tf=90):
        """Run the simulator with the given controller for `tf` seconds, storing results inside this Simulator.

        Args:
            tf (float, optional): how long to run the simulation. Defaults to 90.
        """
        self.log = []
        state = np.zeros(5)
        for t in np.arange(0, tf, 0.01):
            u = self.cb(state)
            assert isinstance(u, np.ndarray), f"expected numpy array from controller but got type {type(u)}"
            assert u.shape==(2,), f"expected shape (2,) from controller but received {u.shape}"
            u = np.array([
                np.clip(u[0], *self._accel_limits),
                np.clip(u[1], *self._steering_vel_limits)
            ])

            if ((state[4] > self._steering_limits[1] and u[1] > 0)
             or (state[4] < self._steering_limits[0] and u[1] < 0)):
                u[1] = 0
            crash = self._check_collision(state)
            slip = self._check_accel(state, u)
            self.log.append((t, state, u, crash, slip))
            state = self.dynamics(state, u).toarray().flatten()
    def get_results(self):
        """get the simulation results. gives a tuple of arrays: (timestamps, states, controls, crash, slip). 
        `crash` is a bool array which is true when the car is colliding with a cone.
        `slip` is a bool array which is true when the car is exceeding friction limits.

        Raises:
            ValueError: if the sim has not been run, there will be no results.

        Returns:
            (array, array, array, array, array): shapes (N,), (5, N), (2, N), (N,), (N,). time series data as described above.
        """
        try: log = self.log
        except: raise ValueError("cannot animate; no results exist. Did you .run() the simulator?")
        ts = np.array([i[0] for i in self.log])
        xs = np.concatenate([i[1][:, np.newaxis] for i in self.log], axis=1)
        us = np.concatenate([i[2][:, np.newaxis] for i in self.log], axis=1)
        crash  = np.array([i[3] for i in self.log])
        slip  = np.array([i[4] for i in self.log])
        return (ts, xs, us, crash, slip)
    def plot(self, block=True):
        """plot the last run of the simulator.

        Args:
            block (bool, optional): the `block` argument to plt.show(). Defaults to True.
        """
        ts, xs, us, crash, slip = self.get_results()
        fig, axs = plt.subplots(7, sharex=True)

        axs[0].plot(ts, xs[0]); axs[0].set_ylabel('x pos (m)')
        axs[1].plot(ts, xs[1]); axs[1].set_ylabel('y pos (m)')
        axs[2].plot(ts, xs[2]); axs[2].set_ylabel('heading (rad)')
        axs[3].plot(ts, xs[3]); axs[3].set_ylabel('velocity (m/s)')
        axs[4].plot(ts, xs[4]); axs[4].set_ylabel('steering angle (rad)')

        axs[5].plot(ts, us[0]); axs[5].set_ylabel('fwd accel (m/s^2)')
        axs[6].plot(ts, us[1]); axs[6].set_ylabel('steering velocity (rad/s)')
        for i in range(5):
            axs[i].scatter(ts[crash], xs[i, crash], color='tab:red', marker='+')
            axs[i].scatter(ts[slip], xs[i, slip], color='tab:orange', marker='x')
        for i in range(2):
            axs[5+i].scatter(ts[crash], us[i, crash], color='tab:red', marker='+')
            axs[5+i].scatter(ts[slip], us[i, slip], color='tab:orange', marker='x')
        plt.show(block=block)
    def animate(self, save=False, filename='sim.gif', block=True):
        """animate the most recent run of the simulator.

        Args:
            save (bool, optional): whether or not to save the animation to a gif. Defaults to False.
            filename (str, optional): filename to save as, if `save` is True. Defaults to 'sim.gif'.
            block (bool, optional): the `block` argument to plt.show(). Defaults to True.
        """
        fig = plt.figure(figsize=(10, 12))
        axs = fig.subplots(2, 1, height_ratios=[4, 1])
        axs[0].set_aspect('equal')
        axs[1].set_aspect(1)
        axs[0].scatter(*self.left_cones.T, color='tab:blue')
        axs[0].scatter(*self.right_cones.T, color='tab:orange')
        outline = axs[0].add_patch(patches.Polygon(self.car_outline, fill=True, closed=True, facecolor='lightblue', edgecolor='black'))
        posearrow = axs[0].add_patch(patches.FancyArrow(0, 0, 1, 0, width=0.1, color='tab:red'))
        accel = axs[1].plot([0],[0])[0]
        axs[0].set_title('car pose')

        axs[1].set_title('net acceleration')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('acceleration (m/s^2)')

        ts, xs, us, crash, slip = self.get_results()
        accel_values = np.array([self._get_accel(x, u) for x, u in zip(xs.T, us.T)])
        axs[1].hlines([12], [0], [np.max(ts)], linestyles='dashed', color='tab:red')
        collision_patches = []
        def frame(i):
            outline_points = ((self.R(xs[2, i])@self.car_outline.T).T + xs[0:2, i])
            arrow_data = dict(
                x=xs[0, i],
                y=xs[1, i],
                dx=np.cos(xs[2, i]),
                dy=np.sin(xs[2, i]),
            )
            if self._check_collision(xs[:, i]):
                collision_car = patches.Polygon(outline_points, color='tab:red', fill=False, linewidth=0.1)
                collision_pose = patches.FancyArrow(**arrow_data, width=0.1, color='black')
                collision_patches.append(axs[0].add_patch(collision_car))
                collision_patches.append(axs[0].add_patch(collision_pose))
            outline.set_xy(outline_points)
            posearrow.set_data(**arrow_data)
            accel.set_xdata(ts[:i+1])
            accel.set_ydata(accel_values[:i+1])
            return [accel, outline, posearrow] + collision_patches

        anim = animation.FuncAnimation(fig, frame, len(ts), interval=10)
        if save:
            anim.save(filename, writer='ffmpeg')
        plt.show(block=block)


if __name__ == '__main__':
    sim = Simulator(lambda x: np.array([1, -0.1*(x[4]-(-0.25))]))
    print(sim.centerline(1.0))
    print(sim.centerline(np.array([1.0, 2.0])))
    print(sim.centerline([1.0, 2.0, 3, 4]))
    sim.run()
    sim.animate()
    sim.plot()
