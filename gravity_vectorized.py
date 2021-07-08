# gravity
import time
import numpy as np
from astropy import constants
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from shapely.geometry import Point
from Utils.plogger import Logger, timed

logformat = '%(asctime)s:%(levelname)s:%(message)s'
Logger.set_logger('gravity.log', logformat, 'INFO')
logger = Logger.getlogger()

G = constants.G.value
EARTH_RADIUS = constants.R_earth.value
EARTH_MASS = constants.M_earth.value
AU = constants.au.value
buffer_radius = 12.0
grid = (50, 50)


class Map:
    @classmethod
    def settings(cls, dimension, figsize):
        cls.fig, cls.ax = plt.subplots(figsize=figsize)
        cls.ax.set_xlim(-1.1*dimension, 1.1*dimension)
        cls.ax.set_ylim(-1.1*dimension, 1.1*dimension)
        cls.fig.suptitle("vectorized")

    @classmethod
    @timed(logger)
    def blip(cls):
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()


class MassObject(Map):
    def __init__(self, mass, x, y, vx, vy, radius, color='black'):
        self._mass = mass
        self._location = Point(x, y)
        self._velocity = Point(vx, vy)
        self._radius = radius
        self._body = mpl_patches.Circle((self._location.x, self._location.y), self._radius, color=color, picker=10)
        self.ax.add_patch(self._body)
        cv_body = self._body.figure.canvas
        cv_body.mpl_connect('pick_event', self.on_pick)
        cv_body.mpl_connect('motion_notify_event', self.on_motion)
        cv_body.mpl_connect('button_release_event', self.on_release)
        self.current_dragging = False
        self._animator = None

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, p):
        self._location = Point(p[0], p[1])
        self._body.center = (self.location.x, self.location.y)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        self._velocity = Point(vel[0], vel[1])

    @property
    def mass(self):
        return self._mass

    @property
    def animator(self):
        return self._animator

    @animator.setter
    def animator(self, animator_instance):
        self._animator = animator_instance

    def on_pick(self, event):
        if event.artist != self._body:
            return
        self.current_dragging = True

    def on_motion(self, event):
        if not self.current_dragging:
            return
        self.location = (event.xdata, event.ydata)
        self._animator.field.remove()
        self._animator.plot_vectorfield()
        self.blip()

    def on_release(self, event):
        self.current_dragging = False
        self.blip()

    def gravity_field(self, x, y, buffer_radius):
        ''' method that returns a tuple of the gravity vectors at meshgrid
            (x, y) due to the MassObject
        '''
        min_radius_2 = (self._radius * buffer_radius)**2
        dx = x - self.location.x
        dy = y - self.location.y
        dx, dy = np.meshgrid(dx, dy)
        radius = dx*dx + dy*dy
        dx = np.where(radius > min_radius_2, dx, np.nan)
        dy = np.where(radius > min_radius_2, dy, np.nan)
        force = - G * self._mass * radius**(-1.5)
        return force * dx, force * dy


class Animation(Map):
    def __init__(self, x, y, mass_objects):
        self.mass_objects = mass_objects
        self.x = x
        self.y = y
        self.fig.canvas.mpl_connect('button_press_event', self.evolve)
        self.evolve_on = False

    # @timed(logger)
    def plot_vectorfield(self):
        x, y = np.meshgrid(self.x, self.y)
        u = np.zeros((len(self.x), 1))
        v = np.zeros((len(self.y), 1))
        u, v = np.meshgrid(u, v)

        for mass_object in self.mass_objects:
            _u, _v = mass_object.gravity_field(self.x, self.y, buffer_radius)
            u += _u
            v += _v

        self.field = self.ax.quiver(x, y, u, v, scale=2)

    def evolve(self, event):
        if not event.dblclick:
            return

        self.evolve_on = not self.evolve_on

        # initiate the matrices
        pos = []
        vel = []
        mass = []
        for mass_object in self.mass_objects:
            pos.append([mass_object.location.x, mass_object.location.y])
            vel.append([mass_object.velocity.x, mass_object.velocity.y])
            mass.append([mass_object.mass])

        pos = np.array(pos)
        vel = np.array(vel)
        mass = np.array(mass)
        acc = self.get_acc(pos, mass, 0.01 * EARTH_RADIUS)
        dt = 1000
        t = 0

        while self.evolve_on:
            if t % 50_000 == 0:
                print(f'time: {t/3600/24:8.1f} days              ', end='\r')

            # (1/2) kick
            vel += acc * dt*0.5

            # drift
            pos += vel * dt

            #update acceleration
            acc = self.get_acc(pos, mass, EARTH_RADIUS)

            # (1/2) kick
            vel += acc * dt*0.5

            # update time
            t += dt

            if t % 20_000 == 0:
                self.update_status(pos, vel)
                self.field.remove()
                self.plot_vectorfield()
                self.blip()

    def update_status(self, pos, vel):
        for index, mass_object in enumerate(self.mass_objects):
            mass_object.location = (pos[index][0], pos[index][1])
            mass_object.velocity = (vel[index][0], vel[index][1])

    @staticmethod
    def get_acc(pos, mass, softening):
        '''
        Calculate the acceleration on each particle due to Newton's Law
        pos  is an N x 3 matrix of positions of mass_objects
        mass is an N x 1 vector of mass_objects
        softening is the softening length
        a is N x 3 matrix of accelerations
        see: https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9
        '''
        # positions r = [x, y] for all mass_objects
        x = pos[:, 0:1]
        y = pos[:, 1:2]

        # matrix that stores all pairwise particle seperations: r_j - r_i
        dx = x.T - x
        dy = y.T - y

        # calculate the acceleration
        inv_r3 = (dx*dx + dy*dy + softening*softening)**(-1.5)
        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        return np.hstack((ax, ay))


def main():
    dimension = AU / 200
    earth_moon = 384_399_000  # distance earth_moon
    dimension = 1.5 * earth_moon
    solar_map = Map()
    solar_map.settings(dimension, (10,10))

    earth = MassObject(
        EARTH_MASS, 0.0, 0.0, +0.0, +0.0, EARTH_RADIUS, color='blue')
    moon = MassObject(
        EARTH_MASS*0.0123, -earth_moon, 0.0, +0.0, -1022, 0.2725*EARTH_RADIUS, color='orange'
    )
    # mars = MassObject(
    #     0.1*EARTH_MASS, dimension*0.2, dimension*0.7, +0, -2_000, 0.75*EARTH_RADIUS, color='red')
    # jupiter = MassObject(
    #     2.0*EARTH_MASS, dimension*0.6, dimension*0.2, 0, 0, 1.4*EARTH_RADIUS, color='green')

    # create one common vector field instance and pass this to each of the mass objects,
    # so if a method of cvf is called from any of the mass objects the result will be the same
    x_vals = np.linspace(-dimension, dimension, grid[0])
    y_vals = np.linspace(-dimension, dimension, grid[1])
    common_animator = Animation(x_vals, y_vals, [earth, moon])
    earth.animator = common_animator
    moon.animator = common_animator
    #.animator = common_animator
    # jupiter.animator = common_animator
    common_animator.plot_vectorfield()
    plt.show()


if __name__ == '__main__':
    main()
