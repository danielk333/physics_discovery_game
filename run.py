import sys
import numpy as np

from game import run


def get_control_params():
    '''These are the parameters that control the "thrust" algorithm. They will appear as float variables in the simulation.
    '''
    params = dict(
        t_start = 1e3,
        force_y = 0,
        force_x = 0,
    )
    return params


def thrust(pos, vel, t, m_wet, **params):
    '''Thruster controller: takes in ship position, velocity, time and the current fuel mass.
    Returns the vector force that the thrust should generate for that time.

    In the keyword arguments "params" are the parameters to control the algorithm.

    When running the actual experiment the parameters used will be fetched from "get_control_params".
    When simulating a mission, the parameters can be changed on the fly since it is a simulation.

    '''
    if t > params['t_start']:
        return np.array([params['force_x'],params['force_y']])
    else:
        return np.array([0.0,0.0])


def force(pos, vel, t, m):
    '''Model of the physics to be used in the simulation, a function of position, velocity, time and mass of the object experiencing the force.
    '''
    return np.array([0.0,0.0])


if __name__=='__main__':
    run(get_control_params, thrust, force, sys.argv)