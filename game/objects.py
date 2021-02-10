#!/usr/bin/env python
""" 

"""


# Import Modules
import numpy as np

import copy
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
    'Thrust',
    'draw_bar',
]

def draw_bar(screen, pos, size, border_color, bar_color, progress):
    '''Based on https://stackoverflow.com/a/54502953 (CC BY-SA 4.0)
    '''

    pg.draw.rect(screen, border_color, (*pos, *size), 1)
    innerPos  = (pos[0]+3, pos[1]+3)
    innerSize = ((size[0]-6) * progress, size[1]-6)
    pg.draw.rect(screen, bar_color, (*innerPos, *innerSize))



class BaseShip(ShipControl):
    """Creates our mission vessel
    """

    def __init__(self, pos, vel, screen_size, steps=10, params = None, thrust_params=None):
        self.pos = pos
        self.vel = vel
        self.screen_size = screen_size

        if params is None:
            self.params = {'m_dry': 0.25, 'm_wet': 1.0, 'dm/F': 1e0}
        else:
            self.params = params

        self.steps = steps

        if thrust_params is None:
            self.thrust_params = self.get_control_params()
        else:
            self.thrust_params = thrust_params

        self._t = 0
        self.F = self.thrust_game_coord(self.pos, self.vel, self._t, **self.thrust_params)


    @property
    def m(self):
        return self.params['m_dry'] + self.params['m_wet']

    def thrust_game_coord(self, pos, vel, t, **params):
        _pos = copy.copy(pos)
        _vel = copy.copy(vel)
        _pos[1] = self.screen_size[1]-pos[1]
        _vel[1] = -vel[1]
        _F = self.thrust(_pos, _vel, t, **params)
        _F[1] = -_F[1]
        return _F

    def force_game_coord(self, pos, vel, t, m):
        _pos = copy.copy(pos)
        _vel = copy.copy(vel)
        _pos[1] = self.screen_size[1]-pos[1]
        _vel[1] = -vel[1]
        _F = self.force(_pos, _vel, t, m)
        _F[1] = -_F[1]
        return _F

    def acceleration(self, landscape):
        if landscape is None:
            F_tot = self.force_game_coord(self.pos, self.vel, self._t, self.m)
        else:
            F_tot = np.array([0.0,0.0])
            for obj in landscape:
                F_ = obj.force(self.pos, self.vel, t=self._t, m=self.m)
                F_tot += F_

        if self.params['m_wet'] > 0:
            self.F = self.thrust_game_coord(self.pos, self.vel, self._t, **self.thrust_params)
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


class Thrust(pg.sprite.Sprite):

    def __init__(self, ship, **kwargs):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("engine.bmp", [0,0,0])
        self.ship = ship

        self.max_force_scale = kwargs.pop('max_force_scale', 1e-4)
        self.max_scale = kwargs.pop('max_scale', 2.0)

        self.original = self.image
        self.original_rect = self.rect

        self.rect.center = ship.rect.center


    def update(self, *args, **kwargs):

        F_norm = np.linalg.norm(self.ship.F)

        scale = F_norm/self.max_force_scale
        if scale > self.max_scale:
            scale = self.max_scale

        theta_F = -np.degrees(np.arctan2(-self.ship.F[1], self.ship.F[0]))+180
        new_size = (int(self.original_rect.width*scale), int(self.original_rect.height*scale))

        _image = pg.transform.scale(self.original, new_size)
        self.image = pg.transform.rotate(_image, -theta_F)

        center = [
            self.ship.rect.center[0] + np.cos(np.radians(theta_F))*self.ship.rect.size[0]*0.5, 
            self.ship.rect.center[1] + np.sin(np.radians(theta_F))*self.ship.rect.size[1]*0.5, 
        ]
        
        self.rect = self.image.get_rect(center=center)


class Ship(pg.sprite.Sprite, BaseShip):

    def __init__(self, pos, vel, screen_size, **kwargs):
        pg.sprite.Sprite.__init__(self)

        self.image, self.rect = load_image("ship.bmp", [0,0,0])

        self.follow_vel = kwargs.pop('follow_vel', False)

        self.engine_sound = load_sound("158894__primeval-polypod__rocket-launch.wav")
        self.engine_sound.set_volume(kwargs.pop('engine_volume', 0.2))

        self.engine_sound_channel = None

        self.original = self.image
        self.engine_on = False

        self.rect.center = pos

        BaseShip.__init__(self, pos, vel, screen_size, **kwargs)


    def done(self, target):
        """returns true if the ship reaches the target"""
        return self.rect.colliderect(target.rect)
        


    def update(self, dt, landscape):
        BaseShip.update(self, dt, landscape)

        F_norm = np.linalg.norm(self.F)

        if F_norm > 1e-9:
            self.engine_on = True
            if self.engine_sound_channel is None:
                self.engine_sound_channel = self.engine_sound.play()
            else:
                if not self.engine_sound_channel.get_busy():
                    self.engine_sound_channel = self.engine_sound.play()
        else:
            self.engine_on = False
            if self.engine_sound_channel is not None:
                if self.engine_sound_channel.get_busy():
                    self.engine_sound.stop()

        if self.follow_vel:
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

        self.frames = [load_image(f"antistar{i}.bmp", [0,0,0]) for i in range(1,4)]
        self.frames += [load_image(f"antistar{i}.bmp", [0,0,0]) for i in range(2,0,-1)]
        self.frame = 0
        self.frame_speed = 10

        self.image, self.rect = load_image("antistar.bmp", [0,0,0])

        screen = pg.display.get_surface()
        self.area = screen.get_rect()
        self.rect.center = pos

        self.pos = pos

        self.m = m


    def update(self, *args, **kwargs):

        self.frame += 1
        if self.frame >= len(self.frames)*self.frame_speed:
            self.frame = 0

        self.image, self.rect = self.frames[int(self.frame//self.frame_speed)]
        self.rect.center = self.pos


    def force(self, pos, vel, **params):
        """
        """
        rv = np.array(self.rect.center) - np.array(pos)
        #exp and repulsive but at a offset
        rv_norm = np.linalg.norm(rv)
        return -1.674e-12*self.m*params['m']*(rv/rv_norm)*np.exp(-(np.log10(self.m)*12 - rv_norm)**2/10.0)


