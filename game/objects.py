#!/usr/bin/env python
""" 

"""


# Import Modules
import numpy as np

import os
import pygame as pg
from pygame.compat import geterror

from functions import load_image, load_sound
from user_input import ShipControl, PhysicsModel

__all__ = [
    'Ship', 
    'SimulatedShip',
    'Target',
    'Star',
    'AntiStar',
    'target_radius',
]

target_radius = 40.0


class BaseShip(ShipControl):
    """Creates our mission vessel
    """

    def __init__(self, pos, vel, steps=10, params = {'m_dry': 0.25, 'm_wet': 1.0, 'dm/F': 1e1}, thrust_params=None):
        self.pos = pos
        self.vel = vel
        self.params = params
        self.steps = steps

        if thrust_params is None:
            self.thrust_params = self.get_control_params()
        else:
            self.thrust_params = thrust_params

        self._t = 0
        self.F = self.thrust(self.pos, self.vel, self._t, **self.thrust_params)


    @property
    def m(self):
        return self.params['m_dry'] + self.params['m_wet']


    def acceleration(self, landscape):
        if landscape is None:
            F_tot = self.force(self.pos, self.vel, self._t, self.m)
        else:
            F_tot = np.array([0.0,0.0])
            for obj in landscape:
                F_ = obj.force(self.pos, self.vel, t=self._t, m=self.m)
                F_tot += F_

        if self.params['m_wet'] > 0:
            self.F = self.thrust(self.pos, self.vel, self._t, **self.thrust_params)
        else:
            self.F = np.array([0.0,0.0])

        F_tot += self.F

        acc = F_tot/self.m

        return acc


    def update(self, dt, landscape):
        """move the fist based on the mouse position

        Uses a multiple step leapfrog symplectic integrator
        """
        
        sub_dt = dt/float(self.steps)
        dm = 0.0

        for ind in range(self.steps):

            acc = self.acceleration(landscape)
            dm += sub_dt*self.params['dm/F']*np.linalg.norm(self.F)

            #kick
            self.vel = [v + a*sub_dt*0.5 for v,a in zip(self.vel, acc)]
            
            #drift
            self.pos = [x + v*sub_dt for x,v in zip(self.pos, self.vel)]

            #kick
            self._t += sub_dt
            acc = self.acceleration(landscape)
            self.vel = [v + a*sub_dt*0.5 for v,a in zip(self.vel, acc)]
            
        #mass reduction
        self.params['m_wet'] -= dm



class Ship(pg.sprite.Sprite, BaseShip):

    def __init__(self, pos, vel, **kwargs):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("ship.bmp", [0,0,0])
        self.engine, self.rect_engine = load_image("engine.bmp", [0,0,0])

        self.original = self.image
        self.original_engine = self.engine

        self.rect.center = pos
        self.rect_engine.center = pos

        BaseShip.__init__(self, pos, vel, **kwargs)


    def done(self, target):
        """returns true if the ship reaches the target"""
        return np.linalg.norm(np.array(self.pos) - np.array(target.rect.center)) < target_radius


    def update(self, dt, landscape):
        BaseShip.update(self, dt, landscape)

        theta = -np.degrees(np.arctan2(self.vel[1], self.vel[0]))
        self.image = pg.transform.rotate(self.original, theta)
        self.rect = self.image.get_rect(center=self.pos)


class SimulatedShip(BaseShip, PhysicsModel):
    pass


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


