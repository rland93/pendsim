import viz
from scipy.signal import cont2discrete
import cvxpy as cp
import numpy as np
from collections import defaultdict
import pandas as pd
import scipy.integrate
import random, datetime
from multiprocessing.dummy import Pool
from matplotlib import pyplot as plt
import math


    'B' : np.array([
            [0.0],
            [1.0 / pend.M],
            [0.0],
            [-1.0 / (pend.l * pend.M)]
        ])
}

class Pendulum(object):
    '''Pendulum object

    Contains pendulum dynamical model and model parameters. 
    '''
    def __init__(self, M, m, l, g=9.81, cfric=0.04, initial_state=np.array([0,0,0,0])):
        '''New pendulum object.

        Parameters
        ----------
        M : float
            Cart mass
        m : float
            Pendulum mass
        l : float
            Pendulum length
        g : float, optional
            Gravitational constant, by default 9.81
        cfric : float, optional
            Cart wheel viscous friction constant, by default 0.04
        initial_state : ndarray of float, shape (4,), optional
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
        t : float
            unused in LTI model, but required by solve_ivp()
        state : array of float, shape (5,)
            5x1 system state with u tacked on at the end:
            [x, xd, theta, thetad, u]

        Returns
        -------
        state : array of float, shape (5,)
            derivative of the state
        '''
        x, xd, theta, thetad, u = state[0], state[1], state[2], state[3], state[4]

        xdd = (u + self.m * self.g * np.sin(theta) * np.cos(theta) \
            - self.m * self.l * thetad * thetad *  np.sin(theta)) \
                / (self.M + self.m - self.m * np.cos(theta) * np.cos(theta))

        tdd = (xdd * np.cos(theta) + self.g * np.sin(theta) ) / self.l

        return np.array([xd * (1 - self.cfric), xdd, thetad, tdd, u])

    def solve(self, dt, system_state, u, solve_args={}):
        '''helper method for solve ivp

        Parameters
        ----------
        dt : float
            [description]
        system_state : np array of float, shape (4,)
            system state given by:
            [x, xd, theta, thetad]
        u : float
            scalar force. Positive is ->, negative is <-
            u is assumed constant throughout the interval dt.
        solve_args : solver arguments, optional
            arguments passed to the solver. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html for
            more details

        Returns
        -------
        tuple of (ndarray, shape (4,), float)
            the state at the end of dt and the force at the end of dt. 
            note that force is held constant through the interval.
        '''
        # roll state and u into a single array
        y = np.empty( (5) )
        y[:4] = system_state
        y[4] = u
        # solve system
        sol = scipy.integrate.solve_ivp(self.system_dynamics, (0, dt), y, **solve_args)
        return sol.y[:4, -1], sol.y[4, -1]

    def get_energy(self, state):
        '''Return 

        [extended_summary]

        Parameters
        ----------
        state : np array of shape (4,)
            system state given by [x, xd, theta, thetad]

        Returns
        -------
        (float, float float)
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

class MPC(object):
    def __init__(self, pend, dt, window):
        self.T = window
        self.u_max = 100
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], sys_disc[1]
        self.costW = np.diag([0, 0, 1, 0])

    def policy(self, state, t, dt, xref, plot=None):
        constr = []
        x = cp.Variable((4, self.T + 1))
        u = cp.Variable((1, self.T))
        for t in range(self.T):
            cost = cp.quad_form(cp.abs(x[:,t+1] - xref), self.costW)
            constr += [x[:, t+1] == x[:,t] + dt * (self.A @ x[:,t] + self.B @ u[:,t])]
            constr += [x[:, 0] == state]
            constr += [cp.abs(u[0,:]) <= self.u_max]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve('ECOS', verbose=True)
        if plot is not None:
            fig, ax, lines = plot
            ax.set_xlim((np.min(list(range(self.T+1))) - 0.5, np.max(list(range(self.T+1))) + 0.1))
            for line in lines:
                if line['type'] == 'plot':
                    xdata = list(range(self.T+1))
                    ydata = np.squeeze(x[line['index'],:].value)
                    line['linesobj'].set_xdata(xdata)
                    line['linesobj'].set_ydata(ydata)

                elif line['type'] == 'hline':
                    y = xref[line['index']]
                    line['linesobj'].set_ydata([y])
                
                elif line['type'] == 'action':
                    xdata = list(range(self.T))
                    ydata = np.squeeze(u.value * 0.01)
                    line['linesobj'].set_xdata(xdata)
                    line['linesobj'].set_ydata(ydata)
            ax.set_ylim((-3, 3))
            fig.canvas.draw()
            fig.canvas.flush_events()

        action = - u[0,0].value
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class Simulation(object):
    def __init__(self, dt, t_final, force):
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.force = force # forcing function
    
    def simulate(self, pendulum, controller, **kwargs):


        # unpack kwargs
        plot = kwargs.pop('plot', False)



        times = []
        t = 0
        state = pendulum.y_0
        datas = defaultdict(list)
        statelabels = ['x', 'xd', 't', 'td']
        setpoints = np.array([
            [0,0,0,0],
            [random.uniform(-1,-1), 0, 0, 0]
        ])
        while t <= self.t_final:
            times.append(t)
            t += self.dt

        if plot:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            l1, = ax.plot(times, [0] * len(times), ls='-', label='x')
            l2, = ax.plot(times, [0] * len(times), ls='-', label='t')
            l3, = ax.plot(times, [0] * len(times), ls='--', label='u')
            h3 = ax.axhline(y=0, color='k', linestyle=':', label='x (setpoint)')
            # future estimates
            lines = [
                {   'linesobj' : l1,
                    'index' : 0,
                    'type'  : 'plot'},
                {   'linesobj' : l2,
                    'index' : 2,
                    'type'  : 'plot'},
                {   'linesobj' : l3,
                    'type'  : 'action',
                    'index' : None},
                {   'linesobj' : h3,
                    'index' : 0,
                    'type'  : 'hline'}
            ]
            ax.legend()
        
        for k, t in enumerate(times):
            data = {}
            force = self.force(t)
            data.update(array_to_kv('state', statelabels , state))
            if t < self.t_final/2:
                setpoint = setpoints[0]
            else:
                setpoint = setpoints[1]

            action, controller_data = controller.policy(
                state,
                t,
                self.dt,
                setpoint,
                plot=(fig, ax, lines)
            )
            data.update(array_to_kv('setpoint', statelabels, setpoint))
            data.update(controller_data)
            data[('energy','kinetic')], data[('energy', 'potential')], data['energy','total'] = pendulum.get_energy(state)
            data[('forces','forces')] = force
            data[('control action','control action')] = action
            force = action
            state, _ = pendulum.solve(self.dt, state, force)
            for k, v in data.items():
                datas[k].append(v)
        if plot:
            plt.ioff()
        return pd.DataFrame(datas, index=times)
    
    def simulate_multiple(self, pendulums, controllers, parallel=True):
        if len(pendulums) != len(controllers):
            raise ValueError('pendulums and controllers must have same length. len(pendulums)={}, len(controllers)={}'.format(len(pendulums), len(controllers)))

        if parallel:
            pool = Pool(16)
            print('Simulating {} runs.'.format(len(pendulums)))
            tic = datetime.now()
            results = pool.starmap(self.simulate, zip(pendulums, controllers))
            toc = datetime.now()
            print('finished in {}'.format(toc - tic))
            return pd.concat(results, axis=0, keys=list(range(len(results))))
        else:
            print('Simulating {} runs.'.format(len(pendulums)))
            tic = datetime.now()
            allresults = []
            for pendulum, controller in zip(pendulums, controllers):
                results = self.simulate(pendulum, controller)
                allresults.append(results)
            toc = datetime.now()
            print('finished in {}'.format(toc - tic))
            return pd.concat(allresults, axis=0, keys = list(range(len(results))))

def array_to_kv(level1_key, level2_keys, array):
    data={}
    if array.shape[0] != len(level2_keys):
        raise(ValueError('Level 2 keys are not same length as array: {} vs {}'.format(level2_keys, array.shape[0])))
    for n, name in enumerate(level2_keys):
        key = (level1_key, name)
        val = array[n]
        data[key] = val
    return data

if __name__ == '__main__':
    dt = 0.05
    pend = Pendulum(
        1.0,
        1.0,
        5.0,
        initial_state=np.array([0,0,0.2,0])
    )
    cont = MPC(
        pend,
        dt,
        30
    )
    sim = Simulation(
        dt,
        10,
        lambda t: 0
    )
    results = sim.simulate(pend, cont, plot=True)
    viz = viz.Visualizer(results, pend, frameskip=1, draw_ghost=True)
    viz.display_viz()