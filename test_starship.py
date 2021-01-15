#!/usr/bin/env python
""" 

based on: pygame.examples.chimp
"""


# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import pygame as pg
from pygame.compat import geterror

if not pg.font:
    print("Warning, fonts disabled")
if not pg.mixer:
    print("Warning, sound disabled")

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")


# functions to create our resources
def load_image(name, colorkey=None):
    fullname = os.path.join(data_dir, name)
    try:
        image = pg.image.load(fullname)
    except pg.error:
        print("Cannot load image:", fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()


def load_sound(name):
    class NoneSound:
        def play(self):
            pass

    if not pg.mixer or not pg.mixer.get_init():
        return NoneSound()
    fullname = os.path.join(data_dir, name)
    try:
        sound = pg.mixer.Sound(fullname)
    except pg.error:
        print("Cannot load sound: %s" % fullname)
        raise SystemExit(str(geterror()))
    return sound


# classes for game objects
class Ship(pg.sprite.Sprite):
    """Creates our mission vessel
    """

    def __init__(self, pos, vel, landscape, steps=10, params = {'m_dry': 0.25, 'm_wet': 1.0, 'dm/F': 1e1}):
        pg.sprite.Sprite.__init__(self)
        
        self.image, self.rect = load_image("ship.bmp", [0,0,0])
        self.engine, self.rect_engine = load_image("engine.bmp", [0,0,0])

        self.original = self.image
        self.original_engine = self.engine

        self.rect.center = pos
        self.rect_engine.center = pos
        self.pos = pos
        self.vel = vel
        self.params = params
        self.steps = steps

        self.landscape = landscape

        self._t = 0
        self.F = self.thrust(self.pos, self.vel, self._t)


    def thrust(self, pos, vel, t):
        if t > 1e3:
            return np.array([0,-0.3e-3])
        else:
            return np.array([0,0])

    @property
    def m(self):
        return self.params['m_dry'] + self.params['m_wet']


    def acceleration(self):
        F_tot = np.array([0.0,0.0])

        for obj in self.landscape:
            F_ = obj.force(self.pos, self.vel, m=self.m)
            F_tot += F_

        if self.params['m_wet'] > 0:
            self.F = self.thrust(self.pos, self.vel, self._t)
        else:
            self.F = np.array([0.0,0.0])

        F_tot += self.F

        acc = F_tot/self.m

        return acc


    def update(self, dt):
        """move the fist based on the mouse position

        Uses a multiple step leapfrog symplectic integrator
        """
        
        sub_dt = dt/float(self.steps)

        for ind in range(self.steps):

            acc = self.acceleration()
            
            #kick
            self.vel = [v + a*sub_dt*0.5 for v,a in zip(self.vel, acc)]
            
            #drift
            self.pos = [x + v*sub_dt for x,v in zip(self.pos, self.vel)]

            #kick
            self._t += sub_dt
            acc = self.acceleration()
            self.vel = [v + a*sub_dt*0.5 for v,a in zip(self.vel, acc)]
            
        #mass reduction
        self.params['m_wet'] -= dt*self.params['dm/F']*np.linalg.norm(self.F)

        theta = -np.degrees(np.arctan2(self.vel[1], self.vel[0]))
        self.image = pg.transform.rotate(self.original, theta)
        self.rect = self.image.get_rect(center=self.pos)



    def done(self, target):
        """returns true if the ship reaches the target"""
        return self.rect.colliderect(target.rect)




class Target(pg.sprite.Sprite):
    """
    """

    def __init__(self, pos):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("target.bmp", [0,0,0])

        screen = pg.display.get_surface()
        self.area = screen.get_rect()
        self.rect.center = pos



class Star(pg.sprite.Sprite):
    """
    """

    def __init__(self, pos, m):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("star.bmp", [0,0,0])

        screen = pg.display.get_surface()
        self.area = screen.get_rect()
        self.rect.center = pos

        self.m = m


    def force(self, pos, vel, **params):
        """
        """
        rv = np.array(self.rect.center) - np.array(pos)
        #r instead of r^2
        return 6.674e-11*self.m*params['m']*rv/np.linalg.norm(rv)**2



class AntiStar(pg.sprite.Sprite):
    """
    """

    def __init__(self, pos, m):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("antistar.bmp", [0,0,0])

        screen = pg.display.get_surface()
        self.area = screen.get_rect()
        self.rect.center = pos

        self.m = m


    def force(self, pos, vel, **params):
        """
        """
        rv = np.array(self.rect.center) - np.array(pos)
        #exp and repulsive but at a offset
        rv_norm = np.linalg.norm(rv)
        return -1.674e-12*self.m*params['m']*(rv/rv_norm)*np.exp(-(np.log10(self.m)*10 - rv_norm)**2/10.0)






def main():
    """this function is called when the program starts.
       it initializes everything it needs, then runs in
       a loop until the function returns."""
    # Initialize Everything

    log = {'t': [], 'pos': [], 'vel': [], 'm': [], 'F': []}

    screen_size = (500, 500)

    pg.init()
    screen = pg.display.set_mode(screen_size)
    pg.display.set_caption("Exploration Mission")
    pg.mouse.set_visible(0)

    # Create The Backgound
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    # Display The Background
    screen.blit(background, (0, 0))
    pg.display.flip()

    # Prepare Game Objects
    clock = pg.time.Clock()
    punch_sound = load_sound("punch.wav")

    landscape = (Star([200,300], 1e10), AntiStar([50,100], 1e10))

    ship = Ship([0,250], [0.5,0.5], landscape)
    target = Target([300, 100])

    allsprites = pg.sprite.RenderPlain((ship, target) + landscape)

    dt = 1.0

    t = 0
    t_max = 4e3*dt

    # Main Loop
    going = True
    while going:
        clock.tick(120)

        t += dt

        log['t'].append(t)
        log['pos'].append([ship.pos[0], screen_size[1] - ship.pos[1]])
        log['vel'].append([ship.vel[0], -ship.vel[1]])
        log['m'].append(ship.m)
        log['F'].append([ship.F[0], -ship.F[1]])

        # Handle Input Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                going = False

        if ship.done(target):
            punch_sound.play()  # punch
            going = False

        allsprites.update(dt)

        # Draw Everything
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
        pg.display.flip()

        if t > t_max:
            going = False


    log_data = np.zeros((len(log['t']), 8))
    log_data[:,0] = np.array(log['t'])
    log_data[:,1] = np.array(log['m'])
    log_data[:,2] = np.array([x[0] for x in log['pos']])
    log_data[:,3] = np.array([x[1] for x in log['pos']])
    log_data[:,4] = np.array([x[0] for x in log['vel']])
    log_data[:,5] = np.array([x[1] for x in log['vel']])
    log_data[:,6] = np.array([x[0] for x in log['F']])
    log_data[:,7] = np.array([x[1] for x in log['F']])

    np.savetxt('log.csv', log_data, delimiter=',', header='t, m, x, y, vx, vy, Fx, Fy')

    pg.quit()


def plot_log():

    header = ['t','m','x','y','vx','vy','Fx', 'Fy']
    log = np.genfromtxt('log.csv', delimiter=',', skip_header=1)

    #total force
    F = np.sqrt(log[:,6]**2 + log[:,7]**2)

    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(3, 3, figure=fig)
 
    ax = fig.add_subplot(gs[0:2, 0:2])
    axes = [
        fig.add_subplot(gs[2,2]),
        fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[1,2]),
        fig.add_subplot(gs[2,0]),
        fig.add_subplot(gs[2,1]),
    ]
    color = 'tab:red'

    axes.append(axes[0].twinx())

    ax.plot(log[:,2], log[:,3])
    
    for i in range(1,6):
        axes[i-1].plot(log[:,0], log[:,i])
        axes[i-1].set_xlabel(header[0])
        axes[i-1].set_ylabel(header[i])

    axes[5].plot(log[:,0], F, color=color)
    axes[5].set_xlabel(header[0])
    axes[5].set_ylabel('F', color=color)



    fig.suptitle("Log data")

    plt.show()


def solve_physics():

    star_pos = np.array([200,200])

    header = ['t','m','x','y','vx','vy','Fx', 'Fy']
    log = np.genfromtxt('log.csv', delimiter=',', skip_header=1)

    x_rel = log[:,2] - star_pos[0]
    y_rel = log[:,3] - star_pos[1]

    r = np.sqrt(x_rel**2 + y_rel**2)

    ax = np.diff(log[:,4])
    ay = np.diff(log[:,5])

    Fx_tot = ax*log[:-1,1]
    Fy_tot = ay*log[:-1,1]

    Fx_ext = Fx_tot - log[1:,6]
    Fy_ext = Fy_tot - log[1:,7]

    F = np.sqrt(log[:,6]**2 + log[:,7]**2)
    F_ext = np.sqrt(Fx_ext**2 + Fy_ext**2)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(2,2,1)
    ax.plot(log[:-1,0], r[:-1])

    ax = fig.add_subplot(2,2,2)
    ax.plot(log[:-1,0], F[:-1])

    ax = fig.add_subplot(2,2,3)
    ax.plot(log[:-1,0], F_ext)

    ax = fig.add_subplot(2,2,4)
    ax.plot(r[:-1], F_ext/log[:-1,1])

    plt.show()



# this calls the 'main' function when this script is executed
if __name__ == "__main__":

    main()
    plot_log()

    solve_physics()