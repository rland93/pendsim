import numpy as np
import math
import pandas as pd
from collections import defaultdict
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.style
from multiprocessing.dummy import Pool
from datetime import datetime
import sys
import controller
from viz import Visualizer

SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 13
plt.rcParams['font.family'] = 'serif'
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('image', cmap='gray')
############## PENDULUM #############
class Pendulum(object):
    '''Inverted pendulum on a cart object

    Parameters
    ----------
    M : float
        Cart mass in kg
    m : float
        Ball mass in kg
    l : float
        Length of connecting rod in m
    g : float, optional
        Acceleration from gravity, by default 9.81
    cfric : float, optional
        Cart viscous friction, by default 0.04
    y_0 : 4x1 array, optional
        Initial position, by default 0,0,0,0:
        initial position 0
        initial cart velocity 0
        initial angle 0
        initial angular velocity 0
    '''
    def __init__(self, M, m, l, g=9.81, cfric=0.04, y_0=np.array([0,0,0,0])):
        
        self.M = M
        self.m = m 
        self.l = l
        self.g = g
        self.cfric = cfric
        self.y_0 = y_0

    def system_dynamics(self, t, y):
        '''System Dynamics of form:
        
        dy/dt = f(t, y)
        y(t_0) = y0

        Parameters
        ----------
        t : float
            scalar time
        y : 5x1 array of float
            state array of the form:

            [x, xdot, theta, thetadot, u], where:
                x is the cart location
                xdot is the cart velocity (+i direction)
                theta is the angle of the pendulum with the cart
                thetadot si the angular velocity of the pendulum with the cart
                u is the force applied to the left side of the pendulum pointed in the (+i) direction

        Returns
        -------
        5x1 array of float
            dy/dt
        '''
        xdd = (y[4] + self.m * self.g * np.sin(y[2]) * np.cos(y[2]) - self.m * self.l * y[3] * y[3] *  np.sin(y[2])) / (self.M + self.m - self.m * np.cos(y[2]) * np.cos(y[2]))
        tdd = (xdd * np.cos(y[2]) + self.g * np.sin(y[2]) ) / self.l
        return np.array([y[1] * (1 - self.cfric), xdd, y[3], tdd, y[4]])

    def solve(self, dt, system_state, u, solve_args):
        '''Solve the system equation over an interval dt, given the system state and input u.

        Parameters
        ----------
        dt : float
            step time interval
        system_state : 4x1 array of float
            the system state at time t
        u : float
            force applied

        Returns
        -------
        tuple of (4x1 array, float)
            the state at t + dt, the force at t + dt
        '''
        # roll state and u into a single array
        y = np.empty( (5) )
        y[:4] = system_state
        y[4] = u
        # solve system
        sol = scipy.integrate.solve_ivp(self.system_dynamics, (0, dt), y, **solve_args)
        return sol.y[:4, -1], sol.y[4, -1]

    def get_energy(self, y):
        '''Get the kinetic & potential energies of the pendulum system 

        Parameters
        ----------
        y : 4x1 array of float
            system state

        Returns
        -------
        tuple of (float, float, float)
            (kinetic energy of the system, potential energy of the system, total energy of the system)
        '''
        # cart vel^2
        v_c2 = y[1] * y[1]
        # pend vel^2
        v_p2 = v_c2 - 2 * self.l * y[1] * y[3] * np.cos(y[2]) + self.l * self.l * y[3] * y[3]
        ke = 0.5 * self.M * v_c2 + 0.5 * self.m * v_p2
        pe = self.m * self.g * self.l * np.cos(y[2])
        return ke, pe, ke + pe

    def get_momentum(self, y):
        '''Get the x-direction momentums of the cart & pendulum system.

        Parameters
        ----------
        x : 4x1 array of float
            system state

        Returns
        -------
        tuple of (float, float, float)
            (cart momentum, pendulum momentum, total momentum) -- in the x-direction
        '''
        p_cart = self.M * y[1] 
        p_pend = self.m * (y[1] + y[3] * np.sin(y[2]))
        return (p_cart, p_pend, p_cart + p_pend)

class Simulation(object):
    '''
    The simulation object. Provide a pendulum, a timestep, a final time, and a list of 
    external forces.
    '''
    def __init__(self, dt, t_final, force, solve_args={}):
        '''New Simulation object.

        Parameters
        ----------
        dt : float
            The time step of the simulation
        t_final : float
            final time
        force : function or array (T/dt, 1)
            function of form func(t) describing the force applied to the pendulum or
            mapping of time to external force applied
        '''
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.force = force
        self.solve_args = solve_args

    def simulate(self, pendulum, controller):
        # initialize
        t_k, y_k, n = 0, pendulum.y_0, 0
        # step time
        datas = defaultdict(list)
        times = []

        xref = lambda t_k: np.array([0.1*t_k, 0, 0, 0])
        while t_k <= self.t_final:
            data = {}
            # print('time={}, x_k={}'.format(round(t_k,3), x_k))
            # forces
            u_k = self.force(t_k)

            tic = datetime.now()

            action, ctrldata = controller.policy(y_k, t_k, self.dt, xref(t_k))
            print(
                'action at {} took {}, theta={}, action={}'.format(
                    str(round(t_k,2)).ljust(4,'0'),
                    datetime.now() - tic,
                    round(y_k[2], 2),
                    round(action, 1),
                )
            )
            data.update(ctrldata)

            # write data returned by controller
            for key, val in data.items():
                data[key] = val
                
            # write data returned by simulation
            for key, val in array_to_kv('state', ['x','xd','t','td'], y_k).items():
                data[key] = val

            data[('forces','forces')] = u_k
            p_cart, p_pend, p_total = pendulum.get_momentum(y_k)
            data[('cart momentum','cart momentum')] = p_cart
            data[('pend momentum','pend momentum')] = p_pend
            data[('total momentum','total momentum')] = p_total
            ke, pe, e = pendulum.get_energy(y_k)
            data[('KE','KE')] = ke
            data[('PE','PE')] = pe
            data[('Energy','Energy')] = e
            data[('control action','control action')] = action
            # build index of times
            for k, v in data.items():
                datas[k].append(v)
            times.append(t_k)

            # add action to extern. force to get total force
            u_k += action
 
            # update state, time
            y_k, _ = pendulum.solve(self.dt, y_k, u_k, self.solve_args)
            t_k += self.dt
            n += 1
            '''
            if n % int(1/self.dt) == 0:
                print(str(round(t_k)), end=' ', flush=True)
            '''
        return pd.DataFrame(datas, index=times)
    
    def simulate_many(self, pendulums, controllers):
        '''run a simulation over `n` pendulums & controllers

        Parameters
        ----------
        pendulums : list, length `n`
            the list of pendulums
        controllers : list, length `n`
            the list of controllers

        Returns
        -------
        pandas DataFrame
            The results. Axis 0 (rows) is a multi-index, with level 0 being the run
            index, and level 1 being each run time. Axis 1 (cols) is each parameter,
            as a multi-index.
        '''
        pool = Pool(16)
        print('Simulating {} runs.'.format(len(pendulums)))
        tic = datetime.now()

        results = pool.starmap(self.simulate, zip(pendulums, controllers))
        toc = datetime.now()
        print('finished in {}'.format(toc - tic))
        return pd.concat(results, axis=0, keys=list(range(len(results))))


def array_to_kv(level1_key, level2_keys, array):
    data={}
    if array.shape[0] != len(level2_keys):
        raise(ValueError('Level 2 keys are not same length as array: {} vs {}'.format(level2_keys, array.shape[0])))
    for n, name in enumerate(level2_keys):
        key = (level1_key, name)
        val = array[n]
        data[key] = val
    return data

def random_force(t, L, theta):
    '''
    Smaller `a` produces more 'jittery' functions.

    '''
    m = int(np.floor(L/theta))
    '''
    f(x) = a_0 + sqrt(2) * sum([
        a_j * cos(2pi j x /L ) + b_j sin(2pi j x /L)
    ]) from 1 to m'''

    a = np.random.normal(0, 1/(2*m+1), (m,2))
    return [np.sqrt(2) + sum([aj[0] *np.cos(2*np.pi * j * x/L) + aj[1] * np.sin(2*np.pi*x/L) for j, aj in enumerate(a)]) for x in t]

def make_single_run_figure(data, show=True, save=False):
    labelmapper = {
        'x' : r'$x$',
        'xd' : r'$\dot{x}$',
        't' : r'$\theta$',
        'td' : r'$\dot{\theta}}$',
    }
    x = data.index
    for i in data['state']:
        fig = plt.figure(tight_layout=False, facecolor='lightgrey')
        state = fig.add_subplot(611, facecolor='white')
        forces = fig.add_subplot(612, sharex=state)
        pred = fig.add_subplot(613, sharex=state)
        errors = fig.add_subplot(614, sharex=state)
        momentum = fig.add_subplot(615, sharex=state)
        energy = fig.add_subplot(616, sharex=state)
        state.grid(True,    which='both', color='lightgrey')
        forces.grid(True,   which='both', color='lightgrey')
        pred.grid(True,     which='both', color='lightgrey')
        errors.grid(True,   which='both', color='lightgrey')
        energy.grid(True,   which='both', color='lightgrey')

        # titles
        state.set_title('State: ' + str(i))
        forces.set_title('Forces')
        pred.set_title(r'Predicted Divergence from Nominal Model ($\mu$)')
        errors.set_title('Prediction Error')
        # x, y labels
        state.set_ylabel(labelmapper[i])
        state.set_xlabel('time (s)')
        forces.set_ylabel('Force (N)')
        pred.set_ylabel(r'$\mu$')
        errors.set_ylabel('Pred. error')
        # data
        state.plot(data.index, data[('state', i)], label =labelmapper[i])
        forces.plot(data.index, data['forces'], label='extern. force')
        forces.plot(data.index, data['control action'], label='actuation', linestyle='--')
        pred.plot(data.index, data[('mu', i)], label=labelmapper[i] + str(', ') + r'$\mu$')
        energy.plot(data.index, data['KE'], label='KE')
        energy.plot(data.index, data['PE'], label='PE')
        energy.plot(data.index, data['Energy'], label='Energy')

        # legends
        state.legend(framealpha=1, facecolor='inherit', loc='best')
        forces.legend(framealpha=1, facecolor='inherit', loc='best')
        errors.legend(framealpha=1, facecolor='inherit', loc='best')
        pred.legend(framealpha=1, facecolor='inherit', loc='best')
        momentum.legend(framealpha=1, facecolor='inherit', loc='best')
        energy.legend(framealpha=1, facecolor='inherit', loc='best')

    plt.show()


if __name__ == '__main__':
    results, labels = 0, 0
    data = pd.concat(
    [
        pd.DataFrame.from_records(np.abs(results['ldiff'].values), columns=labels, index=results.index),
        pd.DataFrame.from_records(np.abs(results['nldiff'].values), columns=labels, index=results.index),
        pd.DataFrame.from_records(np.abs(results['ldiff_n'].values), columns=labels, index=results.index),
        pd.DataFrame.from_records(np.abs(results['nldiff_n'].values), columns=labels, index=results.index),
        pd.DataFrame.from_records(results['state'].values, columns=labels, index=results.index),
        pd.DataFrame.from_records(results['mu'].values, columns=labels, index=results.index),
        pd.DataFrame.from_records(results['sigma'].values, columns=labels, index=results.index),
        pd.DataFrame(results['estimate window'].values, columns=['window'], index=results.index),
        pd.DataFrame(results['forces'].values, columns=['forces'], index=results.index),
        pd.DataFrame(results['KE'].values, columns=['KE'], index=results.index),
        pd.DataFrame(results['PE'].values, columns=['PE'], index=results.index),
        pd.DataFrame(results['Energy'].values, columns=['energy'], index=results.index),
        pd.DataFrame(results['cart momentum'].values, columns=['cart momentum'], index=results.index),
        pd.DataFrame(results['pend momentum'].values, columns=['pendulum momentum'], index=results.index),
        pd.DataFrame(results['total momentum'].values, columns=['total momentum'], index=results.index),
        pd.DataFrame(results['control action'].values, columns=['control action'], index=results.index),
    ],
    axis=1,
    keys = [
        'ldiff', 
        'nldiff', 
        'ldiff_n', 
        'nldiff_n', 
        'state', 
        'mu', 
        'sigma', 
        'window',
        'forces',
        'KE',
        'PE',
        'energy',
        'cart momentum',
        'pend momentum',
        'total momentum',
        'control action',]
    )