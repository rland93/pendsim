import numpy as np
from scipy.integrate import solve_ivp

class Pendulum(object):
    '''Pendulum object

    Contains pendulum dynamical model and model parameters. 
    '''
    def __init__(self, M, m, l, g=9.81, cfric=0.05, initial_state=np.array([0,0,0,0])):
        '''New pendulum object.

        Parameters
        ----------
        M : :obj:`float`
            Cart mass
        m : :obj:`float`
            Pendulum mass
        l : :obj:`float`
            Pendulum length
        g : :obj:`float`, optional
            Gravitational constant, by default 9.81
        cfric : :obj:`float`, optional
            Cart wheel viscous friction constant, by default 0.04
        initial_state : :obj:`np.array` of :obj:`float`, shape (4,), optional
            Initial pendulum state: [x, xdot, theta, thetadot], by default np.array([0,0,0,0]) (up position)
        '''
        self.M = M
        self.m = m 
        self.l = l
        self.g = g
        self.cfric = cfric
        self.y_0 = initial_state
        # jacobians of the system
        # linearized about the upward position x=[0,0,0,0]
        self.jacA = np.array([
            [0, 1.0, 0, 0],
            [0, 0, -self.m * self.g / self.M, 0.0],
            [0, 0, 0, 1.0],
            [0, 0, self.g * (self.m * self.M) / (self.l * self.M), 0.0]
        ])
        self.jacB = np.array([
            [0.0],
            [1.0 / self.M],
            [0.0],
            [-1.0 / (self.l * self.M)]
        ])

    def system_dynamics(self, t, state):
        '''governing equation for pendulum system dynamics, given by:
        
        y' = f(t, y)

        See https://en.wikipedia.org/wiki/Inverted_pendulum for details

        Parameters
        ----------
        t : :obj:`float`
            unused in LTI model, but required by solve_ivp()
        state : :obj:np.array of :obj:`float`, shape (5,)
            5x1 system state with u tacked on at the end:
            [x, xd, theta, thetad, u]

        Returns
        -------
        state : :obj:`np.array` of `float`, shape (5,)
            derivative of the state
        '''
        x, xd, theta, thetad, u = state[0], state[1], state[2], state[3], state[4]

        xdd = (u + self.m * self.g * np.sin(theta) * np.cos(theta) \
            - self.m * self.l * thetad * thetad *  np.sin(theta)) \
                / (self.M + self.m - self.m * np.cos(theta) * np.cos(theta))

        tdd = (xdd * np.cos(theta) + self.g * np.sin(theta) ) / self.l
        return np.array([xd, xdd, thetad, tdd, u])

    def solve(self, dt, system_state, u, solve_args={}):
        '''helper method for solve ivp

        Parameters
        ----------
        dt : :obj:`float`
            [description]
        system_state : :obj:`np.array` of :obj:`float`, shape (4,)
            system state given by:
            [x, xd, theta, thetad]
        u : :obj:`float`
            scalar force. Positive is ->, negative is <-
            u is assumed constant throughout the interval dt.
        solve_args : :obj:`dict`, optional
            arguments passed to the solver. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html for
            more details

        Returns
        -------
        :obj:`tuple` of (:obj:`np.array` of shape (4,), :obj:`float`)
            the state at the end of dt and the force at the end of dt. 
            note that force is held constant through the interval.
        '''
        # roll state and u into a single array
        y = np.empty( (5) )
        y[:4] = system_state
        y[4] = u
        # solve system
        sol = solve_ivp(self.system_dynamics, (0, dt), y, **solve_args)
        return sol.y[:4, -1], sol.y[4, -1]

    def get_energy(self, state):
        '''Return 

        [extended_summary]

        Parameters
        ----------
        state : :obj:`np.array` of shape (4,)
            system state given by [x, xd, theta, thetad]

        Returns
        -------
        :obj:`tuple` of (:obj:`float`, :obj:`float` :obj:`float`)
            kinetic energy, potential energy, total energy
        '''
        x, xd, theta, thetad = state[0], state[1], state[2], state[3]
        # cart vel^2
        v_c2 = xd * xd
        # pend vel^2
        v_p2 = v_c2 - 2 * self.l * xd * thetad * np.cos(theta) + self.l * self.l * thetad * thetad
        # kinetic energy
        ke = 0.5 * self.M * v_c2 + 0.5 * self.m * v_p2
        # potential energy
        pe = self.m * self.g * self.l * np.cos(theta)
        return ke, pe, ke + pe