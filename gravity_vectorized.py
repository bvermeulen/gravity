# gravity
from __future__ import annotations
import numpy as np
from astropy import constants
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import TextBox, Button
from PIL import Image
from shapely.geometry import Point
from Utils.plogger import Logger, timed


logformat = '%(asctime)s:%(levelname)s:%(message)s'
Logger.set_logger('gravity.log', logformat, 'INFO')
logger = Logger.getlogger()

G = constants.G.value
EARTH_RADIUS = constants.R_earth.value
EARTH_MASS = constants.M_earth.value
AU = constants.au.value
earth_moon = 384_399_000
buffer_radius = 10.0   # for solar system use 4.0, for moon use 13.0
grid = (50, 50)
grid = (0, 0) # no vector field shown
magnification = 300  # other the planets get really small
softening = 0.1
rocket_sprite_file = 'rocket_sprite2.png'
degrad = np.pi / 180.0


class Map:
    fig: mpl.figure.Figure
    ax:  mpl.axes.Axes

    @classmethod
    def settings(cls, dimension: float, title: str, figsize: tuple[float, float]):
        cls.fig, cls.ax = plt.subplots(figsize=figsize)
        cls.ax.set_xlim(-1.1*dimension, 1.1*dimension)
        cls.ax.set_ylim(-1.1*dimension, 1.1*dimension)
        cls.fig.suptitle(title)

    @classmethod
    @timed(logger)
    def blit(cls):
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()


class MassObject(Map):
    def __init__(
        self, mass: float, x: float, y: float, vx: float, vy: float,
        radius: float, color: str='black'
        ):
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
        self._animator: Animation

    @property
    def location(self) -> Point:
        return self._location

    @location.setter
    def location(self, p: tuple[float, float]):
        self._location = Point(p[0], p[1])
        self._body.center = (self.location.x, self.location.y)

    @property
    def velocity(self) -> Point:
        return self._velocity

    @velocity.setter
    def velocity(self, vel: tuple[float, float]):
        self._velocity = Point(vel[0], vel[1])

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def animator(self) -> Animation:
        return self._animator

    @animator.setter
    def animator(self, animator_instance: Animation):
        self._animator = animator_instance

    def on_pick(self, event: mpl.backend_bases):
        if event.artist != self._body:
            return
        self.current_dragging = True

    def on_motion(self, event: mpl.backend_bases):
        if not self.current_dragging:
            return
        self.location = (event.xdata, event.ydata)
        self._animator.plot_vectorfield()
        self.blit()

    def on_release(self, _):
        self.current_dragging = False
        self.blit()

    def gravity_field(self, x: np.ndarray, y: np.ndarray, buffer_radius: float) -> tuple[np.ndarray, np.ndarray]:
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


class Rocket(Map):

    rocket_sprite = Image.open(rocket_sprite_file)

    def __init__(self, mass: float, x: float, y: float, velocity: float,
                 alignment: float):
        self._mass = mass
        self._location = Point(x, y)
        self._velocity = Point(0.0, 0.0)
        self._alignment = alignment
        self._delta_v = velocity
        self.handle_delta_v()
        self.rocket = None
        self.add_controls()
        self.update_sprite()

        self.velocitybox.on_submit(self.on_delta_v)
        self.go_button.on_clicked(self.on_go)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def add_controls(self):
        ax_velocitybox = self.fig.add_axes([0.20, 0.05, 0.075, 0.03])
        self.velocitybox = TextBox(ax_velocitybox, 'Delta V:  ')
        self.velocitybox.set_val('+0.0')
        ax_go_button = self.fig.add_axes([0.30, 0.05, 0.05, 0.03])
        self.go_button = Button(ax_go_button, 'GO')
        ax_status_box = self.fig.add_axes([0.38, 0.05, 0.40, 0.03])
        self.statusbox = TextBox(ax_status_box, '')
        self.statusbox.set_val('velocity: 100, delta v: 20%, burn: on')
        self._pause = False
        self._maneuver_flag = False

    @property
    def mass(self):
        return self._mass

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, p: tuple[float, float]):
        self._location = Point(p[0], p[1])
        self.update_sprite()

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel: tuple[float, float]):
        self._velocity = Point(vel[0], vel[1])

    @property
    def maneuver_flag(self):
        return self._maneuver_flag

    @property
    def pause(self):
        return self._pause

    @maneuver_flag.setter
    def maneuver_flag(self, set: bool):
        self._maneuver_flag = set

    def update_sprite(self):
        try:
            self.rocket.remove()

        except AttributeError:
            pass

        im = OffsetImage(
            self.rocket_sprite.rotate(-self._alignment), zoom=0.015
        )
        rocket_im = AnnotationBbox(
            im, (self._location.x, self._location.y), frameon=False
        )
        self.rocket = self.ax.add_artist(rocket_im)
        vel = (self.velocity.x * self.velocity.x + self.velocity.y * self.velocity.y)**0.5
        self.statusbox.set_val(f'velocity: {vel:,.0f}, alignment: {self._alignment:.0f}')

    def on_key(self, event):
        if event.key == 'right':
            self.rotate(1)

        elif event.key == 'left':
            self.rotate(-1)

        elif event.key == ' ':
            self._pause = not self._pause

        elif event.key in ['v', 'V']:
            self.on_go(event)

    def rotate(self, direction: int):
        self._alignment += 4 * direction
        self.update_sprite()
        self.blit()

    def on_delta_v(self, delta_v):
        self._delta_v = float(delta_v)
        print(f'delta v set at: {self._delta_v}')

    def on_go(self, _):
        self.maneuver_flag = True

    def handle_delta_v(self):
        # if there is velocity align burn in the flight direction of the rocket
        # otherwise keep current alignment of thet rocket
        if (self.velocity.y)**2 + (self.velocity.x)**2 < 0.01:
            current_alignment = self._alignment * degrad

        else:
            current_alignment = np.arctan2(self.velocity.x, self.velocity.y)

        new_vel_x = self._velocity.x + np.sin(current_alignment) * self._delta_v
        new_vel_y = self._velocity.y + np.cos(current_alignment) * self._delta_v
        self.velocity = (new_vel_x, new_vel_y)

        # set the alignment of the rocket in the burn direction
        self._alignment = (
            current_alignment / degrad if self._delta_v >= 0
            else (current_alignment / degrad - 180.0) % 360
        )

        # maneuver has been completed
        self.maneuver_flag = False
        print(
            f'\nvelocity: {(self.velocity.x**2 + self.velocity.y**2)**.5:,.0f}, '
            f'delta v: {self._delta_v:,.0f}, alignment: {self._alignment:,.0f}'
        )


class Animation(Map):

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 mass_objects: list[MassObject], rocket: Rocket=None):
        #TODO arguments for blit frequency, dt, etc
        self.mass_objects = mass_objects
        self.x = x
        self.y = y
        self.fig.canvas.mpl_connect('button_press_event', self.evolve)
        self.evolve_on = False
        self.field: mpl.Axes.axes = None
        self.rocket = rocket

    # @timed(logger)
    def plot_vectorfield(self) -> None:
        if len(self.x) < 2 and len(self.y) < 2:
            return

        try:
            self.field.remove()

        except AttributeError:
            pass

        x, y = np.meshgrid(self.x, self.y)
        u = np.zeros((len(self.x), 1))
        v = np.zeros((len(self.y), 1))
        u, v = np.meshgrid(u, v)

        for mass_object in self.mass_objects:
            _u, _v = mass_object.gravity_field(self.x, self.y, buffer_radius)
            u += _u
            v += _v

        self.field = self.ax.quiver(x, y, u, v, scale=2)

    def evolve(self, event: mpl.backend_bases) -> None:
        if not event.dblclick:
            return

        self.evolve_on = not self.evolve_on

        # initiate the matrices
        pos = np.empty((0, 2), np.float64)
        vel = np.empty((0, 2), np.float64)
        mass = np.empty((0, 1), np.float64)
        for mass_object in self.mass_objects:
            pos = np.append(pos,[[mass_object.location.x, mass_object.location.y]], axis=0)
            vel = np.append(vel, [[mass_object.velocity.x, mass_object.velocity.y]], axis=0)
            mass = np.append(mass, [[mass_object.mass]], axis=0)

        if self.rocket:
            pos = np.append(pos, [[self.rocket.location.x, self.rocket.location.y]], axis=0)
            vel = np.append(vel, [[self.rocket.velocity.x, self.rocket.velocity.y]], axis=0)
            mass = np.append(mass, [[self.rocket.mass]], axis=0)

        acc = self.get_acc(pos, mass, softening)
        dt = 1
        t = 0

        while self.evolve_on:

            # print time every hour
            if t % 3600 == 0:
                print(f'time: {t/3600:8.1f} hours           ', end='\r')

            # (1/2) kick method
            vel += acc * dt * 0.5
            pos += vel * dt
            acc = self.get_acc(pos, mass, softening)
            vel += acc * dt * 0.5
            t += dt

            # use 20_000 for solar system, 10_000 for moon, 300 for rocket
            if t % 300 == 0:
                self.plot_vectorfield()
                self.blit()
                self.update_status(pos, vel)
                vel = self.handle_rocket_maneuver(vel)

    def update_status(self, pos: np.ndarray, vel: np.ndarray):
        index = -1
        for index, mass_object in enumerate(self.mass_objects):
            mass_object.location = (pos[index][0], pos[index][1])
            mass_object.velocity = (vel[index][0], vel[index][1])

        if self.rocket:
            self.rocket.velocity = (vel[index+1][0], vel[index+1][1])
            self.rocket.location = (pos[index+1][0], pos[index+1][1])

            while self.rocket.pause:
                self.blit()

    def handle_rocket_maneuver(self, vel: np.ndarray):
        if self.rocket and self.rocket.maneuver_flag:
            self.rocket.handle_delta_v()
            vel[vel.shape[0]-1] = np.array([[self.rocket.velocity.x, self.rocket.velocity.y]])

        return vel


    @staticmethod
    def get_acc(pos: np.ndarray, mass: np.ndarray, softening: float) -> np.ndarray:
        '''
        Calculate the acceleration on each particle due to Newton's Law
        pos  is an N x 2 matrix of positions of mass_objects
        mass is an N x 1 vector of mass_objects
        softening is the softening length
        return:  N x 2 matrix of accelerations
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


def main_moon():
    dimension = 1.5 * earth_moon
    solar_map = Map()
    solar_map.settings(dimension, 'Earth - moon', (10,10))

    earth = MassObject(
        EARTH_MASS, 0.0, 0.0, +0.0, +0.0, EARTH_RADIUS, color='blue')
    moon = MassObject(
        EARTH_MASS*0.0123, -earth_moon, 0.0, +0.0, -1022,
        0.2725*EARTH_RADIUS, color='orange'
    )
    rocket = Rocket(1000, 0.0, 6400_000.0, -25_000.0, -90.0)
    # create one common vector field instance and pass this to each of the mass objects,
    # so if a method of cvf is called from any of the mass objects the result will be the same
    x_vals = np.linspace(-dimension, dimension, grid[0])
    y_vals = np.linspace(-dimension, dimension, grid[1])
    common_animator = Animation(x_vals, y_vals, [earth, moon])
    earth.animator = common_animator
    moon.animator = common_animator
    common_animator.plot_vectorfield()

    plt.show()

def main_rocket():
    ''' Circular velocity of space station at 400 km in orbit is 7,672 m/s
    '''
    dimension = 0.2 * earth_moon
    solar_map = Map()
    solar_map.settings(dimension, 'Earth - rocket', (10,10))

    earth = MassObject(
        EARTH_MASS, 0.0, 0.0, +0.0, +0.0, EARTH_RADIUS, color='blue')
    rocket = Rocket(1000, 0.0, EARTH_RADIUS + 0, 9_000.0, 50.0)
    # create one common vector field instance and pass this to each of the mass objects,
    # so if a method of cvf is called from any of the mass objects the result will be the same
    x_vals = np.linspace(-dimension, dimension, grid[0])
    y_vals = np.linspace(-dimension, dimension, grid[1])
    common_animator = Animation(x_vals, y_vals, [earth], rocket=rocket)
    # earth.animator = common_animator
    common_animator.plot_vectorfield()

    plt.show()

def main_solar():
    dimension = 2 * AU
    solar_map = Map()
    solar_map.settings(
        dimension, 'Solar System - Mercury, Venus, Earth (Moon), Mars', (8, 8))
    sun = MassObject(
        333_000*EARTH_MASS, 0.0, 0.0, +0.0, +0.0,
        15*109*EARTH_RADIUS, color='yellow')
    mercury = MassObject(
        0.055*EARTH_MASS, -0.466697*AU, 0.0, +0.0, -38_860.0,
        magnification*0.3829*EARTH_RADIUS, color='purple')       # at the aphelion
    venus = MassObject(
        0.815*EARTH_MASS, -0.723332*AU, 0.0, +0.0, -35_020.0,
        magnification*0.902*EARTH_RADIUS, color='brown')
    earth = MassObject(
        EARTH_MASS, -AU, 0.0, +0.0, -29_780.0,
        magnification*EARTH_RADIUS, color='blue')
    moon = MassObject(
        0.0123*EARTH_MASS, -(AU + earth_moon), 0.0, +0.0, -(29_780.0+1_022.0),
        2*magnification*0.2725*EARTH_RADIUS, color='orange')
    mars = MassObject(
        0.107*EARTH_MASS, -1.523679*AU, 0.0, +0.0, -24_007.0,
        magnification*0.5333*EARTH_RADIUS, color='red')

    # create one common vector field instance and pass this to each of the mass objects,
    # so if a method of cvf is called from any of the mass objects the result will be the same
    x_vals = np.linspace(-dimension, dimension, grid[0])
    y_vals = np.linspace(-dimension, dimension, grid[1])
    common_animator = Animation(x_vals, y_vals, [sun, mercury, venus, earth, moon, mars])
    sun.animator = common_animator
    mercury.animator = common_animator
    venus.animator = common_animator
    earth.animator = common_animator
    moon.animator = common_animator
    mars.animator = common_animator
    common_animator.plot_vectorfield()
    plt.show()


if __name__ == '__main__':
    main_rocket()
