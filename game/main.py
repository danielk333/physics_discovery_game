#!/usr/bin/env python
""" 

"""


# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import tkinter as tk

import time
import pickle
import sys
import os
import pathlib
import datetime

import pygame as pg
from pygame.compat import geterror

from objects import *
from functions import load_image, load_sound

start_pos, start_vel = [0,250], [0.5,0.5]
target_pos = [300, 110]
dt = 1.0
t_max = 4e3*dt
screen_size = (1024, 768)
my_dpi = 96

done_str = '''
Congratulations! The mission reached the target(s)
★░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░███░██░░░░░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░██░░░█░░░░░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░██░░░██░░░░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░░██░░░███░░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░░░██░░░░██░░░░░░░░░░░░░░░░★ 
★░░░░░░░░░░░██░░░░░███░░░░░░░░░░░░░░★ 
★░░░░░░░░░░░░██░░░░░░██░░░░░░░░░░░░░★ 
★░░░░░░░███████░░░░░░░██░░░░░░░░░░░░★ 
★░░░░█████░░░░░░░░░░░░░░███░██░░░░░░★ 
★░░░██░░░░░████░░░░░░░░░░██████░░░░░★ 
★░░░██░░████░░███░░░░░░░░░░░░░██░░░░★ 
★░░░██░░░░░░░░███░░░░░░░░░░░░░██░░░░★ 
★░░░░██████████░███░░░░░░░░░░░██░░░░★ 
★░░░░██░░░░░░░░████░░░░░░░░░░░██░░░░★ 
★░░░░███████████░░██░░░░░░░░░░██░░░░★ 
★░░░░░░██░░░░░░░████░░░░░██████░░░░░★ 
★░░░░░░██████████░██░░░░███░██░░░░░░★ 
★░░░░░░░░░██░░░░░████░███░░░░░░░░░░░★ 
★░░░░░░░░░█████████████░░░░░░░░░░░░░★ 
★░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░★
'''

def run_sim():

    fig = plt.figure(constrained_layout=True, figsize=(screen_size[0]/my_dpi, screen_size[1]/my_dpi), dpi=my_dpi)

    gs = GridSpec(3, 3, figure=fig)
 
    plane_ax = fig.add_subplot(gs[0:2, 0:2])
    axes = [
        fig.add_subplot(gs[2,2]),
        fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[1,2]),
        fig.add_subplot(gs[2,0]),
        fig.add_subplot(gs[2,1]),
    ]
    color = 'tab:red'

    axes.append(axes[0].twinx())

    title = fig.suptitle("Simulation data")

    targ_pos = np.array(target_pos)

    def update_sim(params):

        ship = SimulatedShip(start_pos, start_vel, thrust_params=params)

        log = {'t': [], 'pos': [], 'vel': [], 'm': [], 'F': []}

        t = 0.0
        # Main Loop
        going = True
        target_hit = False
        while going:
            t += dt

            if np.linalg.norm(np.array(ship.pos) - targ_pos) < target_radius:
                going = False
                target_hit = True

            ship.update(dt, None)

            log['t'].append(t)
            log['pos'].append([ship.pos[0], screen_size[1] - ship.pos[1]])
            log['vel'].append([ship.vel[0], -ship.vel[1]])
            log['m'].append(ship.m)
            log['F'].append([ship.F[0], -ship.F[1]])

            if t > t_max:
                going = False

        if target_hit:
            title.set_text("Simulation data: Target hit!")
        else:
            title.set_text("Simulation data: Target missed")

        for key in log:
            log[key] = np.array(log[key])

        plane_ax.clear()
        for ax in axes:
            ax.clear()

        plane_ax.plot(log['pos'][:,0], log['pos'][:,1])
        
        rect_world = patches.Rectangle((0,0),screen_size[0],screen_size[1],linewidth=1,edgecolor='r',facecolor='none')
        plane_ax.add_patch(rect_world)

        rect_targ = patches.Rectangle(
            (target_pos[0] - target_radius, screen_size[1] - (target_pos[1] - target_radius)),
            target_radius,
            target_radius,
            linewidth=1,edgecolor='g',facecolor='none'
        )
        plane_ax.add_patch(rect_targ)

        axes[0].plot(log['t'], log['m'])
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('m')

        for i, lab_ in zip(range(2), ['x', 'y']):
            axes[1+i].plot(log['t'], log['pos'][:,i])
            axes[1+i].set_xlabel('t')
            axes[1+i].set_ylabel(lab_)            
        for i, lab_ in zip(range(2), ['vx', 'vy']):
            axes[3+i].plot(log['t'], log['vel'][:,i])
            axes[3+i].set_xlabel('t')
            axes[3+i].set_ylabel(lab_)         

        axes[5].plot(log['t'], log['F'][:,0], '--', color=color, label='Fx')
        axes[5].plot(log['t'], log['F'][:,1], ':', color=color, label='Fy')
        axes[5].legend()
        axes[5].set_xlabel('t')
        axes[5].set_ylabel('Thrust', color=color)


    boxes = {}

    ship0 = SimulatedShip(start_pos, start_vel)
    start_params = ship0.get_control_params()

    def update_fig():
        new_params = {}
        for key in boxes:
            try:
                new_params[key] = float(boxes[key].get())
            except:
                print(f'WARNING: input in field {key} failed, using start value')
                new_params[key] = start_params[key]
                boxes[key].delete(0, 'end')
                boxes[key].insert(tk.END, f'{start_params[key]:.4e}')

        print('Updating parameters:')
        for key in new_params:
            print(f'{key}: {new_params[key]}')
        update_sim(new_params)
        plt.draw()

    master = tk.Tk()

    for ind, key in enumerate(start_params):
        tk.Label(master, text=key).grid(row=ind)
        e = tk.Entry(master)
        e.insert(tk.END, f'{start_params[key]:.4e}')
        e.grid(row=ind, column=1)

        boxes[key] = e

    update_sim(start_params)

    tk.Button(
        master, 
        text='Quit', 
        command=master.quit,
    ).grid(
        row=len(start_params), 
        column=0, 
        sticky=tk.W, 
        pady=4,
    )
    tk.Button(
        master, 
        text='Update simulation', 
        command=update_fig,
    ).grid(
        row=len(start_params), 
        column=1, 
        sticky=tk.W, 
        pady=4,
    )

    # plt.draw()
    # tk.mainloop()
    plt.show()


def run_game():
    """this function is called when the program starts.
       it initializes everything it needs, then runs in
       a loop until the function returns."""
    # Initialize Everything

    cnt = pathlib.Path('mission_counter.npy')

    if cnt.is_file():
        with open(cnt, 'rb') as h:
            mission_counter = pickle.load(h)
    else:
        mission_counter = 0

    print(f'{mission_counter} Number of missions performed')

    mission_counter += 1
    with open(cnt, 'wb') as h:
        pickle.dump(mission_counter, h)

    log = {'t': [], 'pos': [], 'vel': [], 'm': [], 'F': []}

    pg.init()
    screen = pg.display.set_mode(screen_size)
    pg.display.set_caption("Exploration Mission")
    pg.mouse.set_visible(1)

    font = pg.font.SysFont(None, 24)

    # Create The Backgound
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    # Display The Background
    screen.blit(background, (0, 0))
    pg.display.flip()

    # Prepare Game Objects
    clock = pg.time.Clock()
    win = load_sound("448274__henryrichard__sfx-success.wav")
    warp = load_sound("453391__breviceps__warp-sfx.wav")

    music = [
        "Kai_Engel-Brand_New_World.wav",
        "Jason_Shaw-Tech_Talk.wav",
        "Jason_Shaw-Vanishing_Horizon.wav",
    ]

    pg.mixer.init()
    pg.mixer.music.load('data/' + music[0])

    ## put landscape here ##
    landscape = (Star([200,300], 1e10), AntiStar([50,100], 1e10))
    ## landscape def end ##

    ship = Ship(start_pos, start_vel)
    ship._layer = 2

    engine = Thrust(ship)
    target = Target(target_pos)

    sprites = pg.sprite.LayeredUpdates((ship, target) + landscape, default_layer=0)

    t = 0
    # Main Loop
    going = True
    pg.mixer.music.play()

    time0 = time.time()
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

        if ship.engine_on:
            sprites.add(engine, layer=1)
        else:
            if engine in sprites:
                sprites.remove(engine)

        if ship.done(target):
            print(done_str)
            
            pg.mixer.music.stop()

            channel = win.play()
            while channel.get_busy():
                pg.time.wait(100)  # ms

            alpha = 0
            num = 100

            channel = warp.play()
            while channel.get_busy():
                pg.time.wait(100)  # ms

            going = False

        sprites.update(dt, landscape)

        # Blit background
        screen.blit(background, (0, 0))

        texts = []

        #Draw counter
        texts.append([
            font.render(f'Mission number: {mission_counter}', True, (100, 100, 255)), 
            (700, 20),
        ])
        
        #Draw timer
        time_elapsed = time.time() - time0
        texts.append([
            font.render(str(datetime.timedelta(seconds=time_elapsed)).split('.')[0], True, (100, 100, 255)),
            (700, 40),
        ])

        #Draw counter
        if ship.engine_on:
            texts.append([
                font.render(f'Engine status: On', True, (100, 255, 100)),
                (700, 60),
            ])
            texts.append([
                font.render(f'Fx = {ship.F[0]:.2e}', True, (100, 255, 100)),
                (700, 80),
            ])
            texts.append([
                font.render(f'Fy = {-ship.F[1]:.2e}', True, (100, 255, 100)),
                (700, 100),
            ])
        else:
            texts.append([
                font.render(f'Engine status: Off', True, (255, 100, 100)),
                (700, 60),
            ])

        for txt, txt_pos in texts:
            screen.blit(txt, txt_pos)

        # Draw "Everything"
        sprites.draw(screen)

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

    np.savetxt(f'log_mission_{mission_counter}.csv', log_data, delimiter=',', header='t, m, x, y, vx, vy, Fx, Fy')

    pg.quit()


def plot_log(log_num):
    fname = f'log_mission_{log_num}.csv'

    if not pathlib.Path(fname).is_file():
        raise ValueError(f'Log for mission number {log_num} does not exist')

    header = ['t','m','x','y','vx','vy','Fx', 'Fy']
    log = np.genfromtxt(fname, delimiter=',', skip_header=1)

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


    #insert other convenience functions like fitting arbitrary functions ect ect
    

    plt.show()



# this calls the 'main' function when this script is executed
if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError('Needs at lest one argument, use command "help" to see commands')

    arg = sys.argv[1].lower().strip()

    if arg == 'help':
        print('Available commands:')
        print('- log [num]: Quicklook plot of the logfile from mission number [num]')
        print('- sim      : Simulate the currently modeled physics in the "force" and "thrust" functions')
        print('- exp      : Run the experiment with the "thrust" function in the real world physics')
    elif arg == 'log':
        if len(sys.argv) < 3:
            logn = 0
        else:
            logn = int(sys.argv[2])
        plot_log(logn)
    elif arg == 'sim':
        run_sim()
    elif arg == 'exp':
        run_game()
    else:
        raise ValueError('Invalid command, use "help" to see valid commands')
    