import numpy as np

class ShipControl:

    def get_control_params(self):
        '''These are the parameters that control the "thrust" algorithm. They will appear as float variables in the simulation.
        '''
        params = dict(
            t_start = 1e3,
            force_y = -1e-4,
            force_x = 0,
        )
        return params

    def thrust(self, pos, vel, t, **params):
        '''Thruster controller: takes in ship position, velocity and time.
        Returns the vector force to thrust for that time.

        In the keyword arguments "params" are the parameters to control the algorithm.

        When running the actual experiment the parameters used will be fetched from "get_params".
        When simulating a mission, the parameters can be changed on the fly since it is a simulation.

        '''
        if t > params['t_start']:
            return np.array([params['force_x'],params['force_y']])
        else:
            return np.array([0.0,0.0])

class PhysicsModel:

    def force(self, pos, vel, t, m):
        return np.array([0.0,0.0])