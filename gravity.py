# gravity
import numpy as np
from astropy import constants
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from shapely.geometry import Point

G = constants.G.value
EARTH_RADIUS = constants.R_earth.value
EARTH_MASS = constants.M_earth.value
AU = constants.au.value
buffer_radius = 10.0


class Map:
    @classmethod
    def settings(cls, dimension, figsize):
        cls.fig, cls.ax = plt.subplots(figsize=figsize)
        cls.ax.set_xlim(-0.1*dimension, 1.1*dimension)
        cls.ax.set_ylim(-0.1*dimension, 1.1*dimension)

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
        self.blip()

    def on_release(self, event):
        self.cvf.field.remove()
        self.cvf.plot_vectorfield()
        self.current_dragging = False
        self.blip()

    def grav_vec(self, x, y, buffer_radius):
        ''' method that returns a tuple of the gravity vector at location
            (x, y) due to the MassObject
        '''
        dx = (x - self.location.x)
        dy = (y - self.location.y)
        radius = np.sqrt(dx*dx + dy*dy)
        if radius < buffer_radius * self.radius:
            return np.nan, np.nan

        angle = np.arctan2(dy, dx)
        force = - G * self.mass / (radius * radius)
        return force * np.cos(angle), force * np.sin(angle)


class VectorField(Map):
    def __init__(self, x, y, vector_fields):
        self.vector_fields = vector_fields
        self.x_vals = x
        self.y_vals = y

    def plot_vectorfield(self):
        u = [0 for _ in range(len(self.y_vals)*len(self.x_vals))]
        v = u.copy()
        for vector_field in self.vector_fields:
            index = 0
            for y in self.y_vals:
                for x in self.x_vals:
                    vector = vector_field(x, y, buffer_radius)
                    if np.isnan(vector[0]):
                        u[index] = np.nan
                        v[index] = np.nan

                    else:
                        u[index] += vector[0]
                        v[index] += vector[1]

                    index += 1

        u = np.array(u)
        v = np.array(v)
        x, y = np.meshgrid(self.x_vals, self.y_vals)
        self.field = self.ax.quiver(x, y, u, v, scale=2)


def main():
    dimension = AU / 200
    x_vals = np.linspace(0, dimension, 39)
    y_vals = np.linspace(0, dimension, 39)
    solar_map = Map()
    solar_map.settings(dimension, (10,10))

    earth = MassObject(
        EARTH_MASS, dimension*0.6, dimension*0.7, EARTH_RADIUS, color='blue')
    mars = MassObject(
        0.3*EARTH_MASS, dimension*0.2, dimension*0.7, 0.8*EARTH_RADIUS, color='red')
    jupiter = MassObject(
        2.0*EARTH_MASS, dimension*0.6, dimension*0.2, 1.4*EARTH_RADIUS, color='green')

    # create one common vector field and pass to each of the mass objects
    cvf = VectorField(x_vals, y_vals, [mars.grav_vec, earth.grav_vec, jupiter.grav_vec])
    earth.cvf = cvf
    mars.cvf = cvf
    jupiter.cvf = cvf
    cvf.plot_vectorfield()
    plt.show()


if __name__ == '__main__':
    main()
