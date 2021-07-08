# gravity
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
buffer_radius = 10.0
grid = (100, 100)


class Map:
    @classmethod
    def settings(cls, dimension, figsize):
        cls.fig, cls.ax = plt.subplots(figsize=figsize)
        cls.ax.set_xlim(-0.1*dimension, 1.1*dimension)
        cls.ax.set_ylim(-0.1*dimension, 1.1*dimension)
        cls.fig.suptitle("vectorized")

    @classmethod
    def blip(cls):
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()


class MassObject(Map):
    def __init__(self, mass, x, y, radius, color='black'):
        self.mass = mass
        self.location = Point(x, y)
        self.radius = radius
        self.body = mpl_patches.Circle((self.location.x, self.location.y), self.radius, color=color, picker=10)
        self.ax.add_patch(self.body)
        cv_body = self.body.figure.canvas
        cv_body.mpl_connect('pick_event', self.on_pick)
        cv_body.mpl_connect('motion_notify_event', self.on_motion)
        cv_body.mpl_connect('button_release_event', self.on_release)
        self.current_dragging = False
        self.cvf = None

    def on_pick(self, event):
        if event.artist != self.body:
            return
        self.current_dragging = True

    def on_motion(self, event):
        if not self.current_dragging:
            return
        self.location = Point(event.xdata, event.ydata)
        self.body.center = (self.location.x, self.location.y)
        self.cvf.field.remove()
        self.cvf.plot_vectorfield()
        self.blip()

    def on_release(self, event):
        # self.cvf.field.remove()
        # self.cvf.plot_vectorfield()
        self.current_dragging = False
        self.blip()

    def grav_vec(self, x, y, buffer_radius):
        ''' method that returns a tuple of the gravity vectors at meshgrid
            (x, y) due to the MassObject
        '''
        min_radius_2 = (self.radius * buffer_radius)**2
        dx = x - self.location.x
        dy = y - self.location.y
        dx, dy = np.meshgrid(dx, dy)
        radius = dx*dx + dy*dy
        dx = np.where(radius > min_radius_2, dx, np.nan)
        dy = np.where(radius > min_radius_2, dy, np.nan)
        force = - G * self.mass * radius**(-1.5)
        return force * dx, force * dy


class VectorField(Map):
    def __init__(self, x, y, vector_fields):
        self.vector_fields = vector_fields
        self.x = x
        self.y = y

    @timed(logger)
    def plot_vectorfield(self):
        x, y = np.meshgrid(self.x, self.y)
        u = np.zeros((len(self.x), 1))
        v = np.zeros((len(self.y), 1))
        u, v = np.meshgrid(u, v)

        for vector_field in self.vector_fields:
            _u, _v = vector_field(self.x, self.y, buffer_radius)
            u += _u
            v += _v

        self.field = self.ax.quiver(x, y, u, v, scale=2)


def main():
    dimension = AU / 200
    solar_map = Map()
    solar_map.settings(dimension, (10,10))

    earth = MassObject(
        EARTH_MASS, dimension*0.6, dimension*0.7, EARTH_RADIUS, color='blue')
    mars = MassObject(
        0.3*EARTH_MASS, dimension*0.2, dimension*0.7, 0.8*EARTH_RADIUS, color='red')
    jupiter = MassObject(
        2.0*EARTH_MASS, dimension*0.6, dimension*0.2, 1.4*EARTH_RADIUS, color='green')

    # create one common vector field instance and pass this to each of the mass objects,
    # so if a method of cvf is called from any of the mass objects the result will be the same
    x_vals = np.linspace(0, dimension, grid[0])
    y_vals = np.linspace(0, dimension, grid[1])
    cvf = VectorField(x_vals, y_vals, [earth.grav_vec, mars.grav_vec, jupiter.grav_vec])
    earth.cvf = cvf
    mars.cvf = cvf
    jupiter.cvf = cvf
    cvf.plot_vectorfield()
    plt.show()


if __name__ == '__main__':
    main()
