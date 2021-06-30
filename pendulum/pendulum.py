import numpy as np
from scipy.integrate import solve_ivp

class Pendulum(object):
    '''Pendulum object

    Contains pendulum dynamical model and model parameters. 
    '''
    def __init__(self, M, m, l, g=9.81, cfric=0.1, pfric=.05, initial_state=np.array([0,0,0,0])):
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
            Cart wheel viscous friction constant, by default 0.1
        pfric : :obj:`float`, optional
            pendulum bearing viscous friction constant, by default 0.05
        initial_state : :obj:`np.array` of :obj:`float`, shape (4,), optional
            Initial pendulum state: [x, xdot, theta, thetadot], by default np.array([0,0,0,0]) (up position)
        '''
        self.M = M
        self.m = m 
        self.l = l
        self.g = g
        self.cfric = cfric
        self.pfric = pfric
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
        # unpack variables
        xd, t, td, u = state[1], state[2], state[3], state[4]
        m, M, l, g, cfric, pfric = self.m, self.M, self.l, self.g, self.cfric, self.pfric
        cost, sint = np.cos(t), np.sin(t)
        
        # solve for derivatives
        xdd = (g*m*sint*cost+u-m*l*td*td*sint)/(M+m-m*cost*cost)
        tdd = (g*sint+xdd*cost)/l
        
        # frictions
        xdd += - cfric*xd
        tdd += - pfric*td

        return np.array([xd, xdd, td, tdd, u])

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
        xd, t, td = state[1], state[2], state[3]

        ke = 0.5 * (self.m+self.M) * xd * xd - \
            self.m * self.l * xd * td * np.cos(t) + \
                0.5 * self.m * self.l * self.l * td * td
        # potential energy
        pe = self.m * self.g * self.l * np.cos(t)
        return ke, pe, ke + pe