
# Import Modules
import os
import json
import pickle
import pygame as pg
from pygame.compat import geterror
import sys
import pathlib
import configparser

from .objects import *

def get_path(name):
    return os.path.join(data_dir, name)


def get_config_level_conf(config, section):
    level_conf = {}
    level_conf['star'] = {}
    level_conf['antistar'] = {}
    for key in config[section]:
        parts = key.split('_')
        if parts[0] in level_conf:
            if parts[1] not in level_conf[parts[0]]:
                level_conf[parts[0]][parts[1]] = {}

            level_conf[parts[0]][parts[1]][parts[2]] = json.loads(config.get(section, key))

    return level_conf



def get_save_data():
    dat = pathlib.Path(get_path('save_data.npy'))

    if dat.is_file():
        with open(dat, 'rb') as h:
            mission_counter, level_number = pickle.load(h)
    else:
        mission_counter = 0
        level_number = 1

    return mission_counter, level_number

def put_save_data(mission_counter, level_number):
    dat = pathlib.Path(get_path('save_data.npy'))

    with open(dat, 'wb') as h:
        pickle.dump([mission_counter, level_number], h)


def get_config(pth):
    config = configparser.ConfigParser(interpolation=None)
    conffile = pathlib.Path(get_path(pth))

    if conffile.exists() and conffile.is_file():
        config.read([conffile])
    else:
        raise ValueError('No game config file found')

    return config


if not pg.font:
    print("Warning, fonts disabled")
if not pg.mixer:
    print("Warning, sound disabled")

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")


def load_image(name, colorkey=None):
    '''Based on pygame documented examples.
    '''

    fullname = get_path(name)
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
    '''Based on pygame documented examples.
    '''

    class NoneSound:
        def play(self):
            pass

    if not pg.mixer or not pg.mixer.get_init():
        return NoneSound()
    fullname = get_path(name)
    try:
        sound = pg.mixer.Sound(fullname)
    except pg.error:
        print("Cannot load sound: %s" % fullname)
        raise SystemExit(str(geterror()))
    return sound

