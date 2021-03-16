import numpy as np
import pandas as pd
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.style
matplotlib.style.use('seaborn')
import pathos.pools
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

############## PENDULUM #############

class Pendulum(object):
    '''
    An inverted pendulum object.
    '''
    def __init__(self, M, m, l, g, x_0=np.array([0,0,0,0]), cfric=0.0075, pfric=0.0075):
        '''
        M: cart mass
        m: ball mass
        l: pend length
        g: gravity
        cfric: cart (viscous) friction
        pfric: pend (viscout) friction
        x_0: initial state
        '''
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.cfric = cfric
        self.pfric = pfric
        self.x_0 = x_0

    def pend_eqn(self, t, x):
        '''
        Calculate xdot = f(x) where x is the vector
        [x, xdot, t, tdot, u]
             |
        [xd, xdd, td, tdd, u]
        '''
        g = self.g
        M = self.M
        m = self.m
        l = self.l
        u = x[4]
        xd = x[1] * (1-self.cfric)
        td = x[3]
        sint = np.sin(x[2])
        cost = np.cos(x[2])
        xdd = (u + m*g*sint*cost - m*l*td*td*sint) / (M+m - m*cost*cost)
        tdd = ( xdd*cost + g*sint ) / l

        return np.array([xd, xdd, td, tdd, u])

    def solve(self, pend_eqn, dt, state):
        sol = scipy.integrate.solve_ivp(pend_eqn, (0, dt), state)
        print(sol)
        final_y = sol.y[:4, -1]
        return final_y

    def get_energy(self, x):
        '''
        Get a tuple containing the kinetic and potential energy of the system.
        '''
        # cart vel^2
        v_c2 = x[1] * x[1]
        # pend vel^2
        v_p2 = v_c2 - 2 * self.l * x[1] * x[3] * np.cos(x[2]) + self.l * self.l * x[3] * x[3]
        ke = 0.5 * self.M * v_c2 + 0.5 * self.m * v_p2
        pe = self.m * self.g * self.l * np.cos(x[2])
        return ke, pe, ke + pe

    def get_momentum(self, x):
        '''
        Calculate the (x-direction) linear momentum of the system.
        '''
        p_cart = self.M * x[1] 
        p_pend = self.m * (x[1] + x[3] * np.sin(x[2]))
        return p_cart, p_pend

class Simulation(object):
    '''
    The simulation object. Provide a pendulum, a timestep, a final time, and a list of 
    external forces.
    '''
    def __init__(self, pend, dt, t_final, force, noise_scale):
        self.pend = pend # pendulum to be simulated
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.force = force # forcing function
        self.noise_scale = noise_scale # noise given to the state. can be scalar (equal noise) or len 4 array (noise given to each state)

    def simulate(self, controller):
        '''
        Run the simulation with the given controller
        '''
        # initialize
        t_k = 0
        x_k = self.pend.x_0
        # no of runs
        n = 0

        # run the policy once to see what data it returns, then populate run_data with that
        _, sample = controller.policy(x_k, t_k, self.dt)
        run_data = {}
        for k, v in sample.items():
            run_data[k] = []
        
        run_data['state'] = []
        run_data['forces'] = []
        run_data['cart momentum'] = []
        run_data['pend momentum'] = []
        run_data['total momentum'] = []
        run_data['KE'] = []
        run_data['PE'] = []
        run_data['Energy'] = []
        run_data['control action'] = []
        run_data['estimate window'] = []
        times = []

        
        # step time
        while t_k <= self.t_final:
            # print('time={}, x_k={}'.format(round(t_k,3), x_k))
            # forces
            u_k = self.force(t_k)

            action, data = controller.policy(x_k, t_k, self.dt)
            # write data returned by controller
            for key, val in data.items():
                run_data[key].append(val.flatten())
                
            # write data returned by simulation
            times.append(t_k)
            run_data['state'].append(x_k.flatten())
            run_data['forces'].append(u_k)
            p_cart, p_pend = self.pend.get_momentum(x_k)
            run_data['cart momentum'].append(p_cart)
            run_data['pend momentum'].append(p_pend)
            run_data['total momentum'].append(p_cart + p_pend)
            ke, pe, e = self.pend.get_energy(x_k)
            run_data['KE'].append(ke)
            run_data['PE'].append(pe)
            run_data['Energy'].append(e)
            run_data['control action'].append(action)
            run_data['estimate window'].append(controller.M)

            # add action to extern. force to get total force
            u_k += action

            solve_input = np.empty(dtype=float, shape=(5))
            solve_input[:4] = x_k
            solve_input[4] = u_k
 
            # update state, time
            x_k = self.pend.solve(self.pend.pend_eqn, self.dt, solve_input)
            t_k += self.dt
            n += 1
        
        return pd.DataFrame(data=run_data, index=times)

class simRunner(object):
    def __init__(self):
        self.run_params = []

    def run_once(self, run_consts):
        pend_const, sim_const, ctrl_const = run_consts
        '''Run the sim one time'''
        # Pend parameters
        ########################################
        M = pend_const['M']
        m = pend_const['m']
        l = pend_const['l']
        g = pend_const['g']
        init = pend_const['init']
        pendulum = Pendulum(M, m, l, g, x_0=init, cfric=0.0075, pfric=0.0075)

        # Force
        ########################################
        if sim_const['force']:
            force = sim_const['force']
        else:
            force = lambda t: 0
        
        # Enter Values
        ########################################
        sim = Simulation(
            pendulum,
            sim_const['dt'],
            sim_const['simtime'],
            force,
            sim_const['noise']
        )
        ctrl = controller.MPCWithGPR(
            pendulum,
            sim_const['dt'],
            ctrl_const['measure_n'],
            ctrl_const['window'],
        )
        # Run Simulation
        ########################################
        print('.', end='')
        sys.stdout.flush()
        simdata = sim.simulate(ctrl)

        simdata['ldiff'] = simdata['state'].shift(-1) - simdata['lpred']
        simdata['nldiff'] = simdata['state'].shift(-1) - simdata['nlpred']
        simdata['ldiff_n'] = simdata['state'].shift(-ctrl_const['measure_n']) - simdata['lpred_n']
        simdata['nldiff_n'] = simdata['state'].shift(-ctrl_const['measure_n']) - simdata['nlpred_n']

        return simdata

    def set_params(self, params, random):
        # mode 1 = random pendulums per run
        if random:
            pend = {
                'M' : np.random.uniform(low=params['M_low'], high=params['M_high']),
                'm' : np.random.uniform(low=params['m_low'], high=params['m_high']),
                'l' : np.random.uniform(low=params['l_low'], high=params['l_high']),
                'g' : 9.81,
                'init' : np.array([0,0,0,0]),
            }
            sim = {
                'dt' : 0.001,
                'simtime' : params['simtime'],
                'force' : params['force'],
                'noise' : 0,
            }
            ctrl = {
                'window' : params['window'],
                'measure_n' : params['measure_n']
            }
        else:
            pend = {
                'M' : params['M'],
                'm' : params['m'],
                'l' : params['l'],
                'g' : 9.81,
                'init' : np.array([0,0,0,0])
            }
            sim = {
                'dt' : 0.001,
                'simtime' : params['simtime'],
                'force' : params['force'],
                'noise' : 0
            }
            ctrl = {
                'window' : params['window'],
                'measure_n' : params['measure_n'],
            }
        self.run_params.append((pend, sim, ctrl))
        return pend, sim, ctrl
    
    @staticmethod
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



        # return [a[0,0] + np.sqrt(2) * sum([a_j[0] * np.cos(2*np.pi*j*x/L) + a_j[1] * np.sin(2*np.pi*j*x/L) for j, a_j in enumerate(a)]) for x in t]



    def run_many(self, runs, params, random):
        paramlist = []
        for r in range(runs):
            paramlist.append(self.set_params(params, random))
        
        p = pathos.pools.ProcessPool()
        print('simulating {} runs'.format(runs))
        tic = datetime.now()
        results = p.map(self.run_once, paramlist)
        toc = datetime.now()
        print('finished in {}'.format(toc - tic))
        level_list = list(range(len(results)))
        return pd.concat(results, keys=level_list, axis=0)


def make_single_run_figure(data, path_prefix='./', save=False, show=False):
    x = data.index
    labels = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
    

    for i in data['state']:
        fig = plt.figure(figsize=(7.5,10), tight_layout=True)
        state = fig.add_subplot(611)
        forces = fig.add_subplot(612, sharex=state)
        pred = fig.add_subplot(613, sharex=state)
        errors = fig.add_subplot(614, sharex=state)
        momentum = fig.add_subplot(615, sharex=state)
        energy = fig.add_subplot(616, sharex=state)
        state.grid(True,    which='both', color='lightgrey')
        forces.grid(True,   which='both', color='lightgrey')
        pred.grid(True,     which='both', color='lightgrey')
        errors.grid(True,   which='both', color='lightgrey')
        momentum.grid(True, which='both', color='lightgrey')
        energy.grid(True, which='both', color='lightgrey')

        # titles
        state.set_title('State: ' + str(i))
        forces.set_title('Forces')
        pred.set_title(r'Predicted Divergence from Nominal Model ($\mu$)')
        errors.set_title('Prediction Error')
        # x, y labels
        state.set_ylabel(i)
        state.set_xlabel('time (s)')
        forces.set_ylabel('Force (N)')
        forces.set_xlabel('time (s)')
        pred.set_ylabel(r'$\mu$')
        pred.set_xlabel('time (s)')
        errors.set_ylabel('Pred. error')
        errors.set_xlabel('time (s)')
        # data

        state.plot(data.index, data[('state', i)], label = i)
        forces.plot(data.index, data['forces'], label='extern. force')
        forces.plot(data.index, data['control action'], label='actuation', linestyle='--')
        pred.plot(data.index, data[('mu', i)])
        errors.plot(data.index, data[('ldiff', i)], label='ldiff')
        errors.plot(data.index, data[('nldiff', i)], label='nldiff')
        momentum.plot(data.index, data['cart momentum'], label='cart momentum')
        momentum.plot(data.index, data['pend momentum'], label='pend momentum')
        momentum.plot(data.index, data['total momentum'], label='total momentum')
        energy.plot(data.index, data['KE'], label='KE')
        energy.plot(data.index, data['PE'], label='PE')
        energy.plot(data.index, data['energy'], label='Energy')

        # legends
        state.legend(framealpha=1, facecolor='inherit', loc='best')
        forces.legend(framealpha=1, facecolor='inherit', loc='best')
        errors.legend(framealpha=1, facecolor='inherit', loc='best')
        pred.legend(framealpha=1, facecolor='inherit', loc='best')
        momentum.legend(framealpha=1, facecolor='inherit', loc='best')
        energy.legend(framealpha=1, facecolor='inherit', loc='best')

    plt.show()


if __name__ == '__main__':
    a = 0.161
    b = 1.0
    c = 10
    pend_const = {
        'M' : 4,
        'm' : 2,
        'l' : 3,
        'g' : 9.81,
        'init' : np.array([0,0,0.1,0])
    }
    sim_const = {
        'dt' : 0.001,
        'simtime' : 10,
        'force' : lambda t: c * 1/abs(a*np.pi) * np.exp( -((t-b)/a)**2 ),
        'noise' : 0
    }
    ctrl_const = {
        'window' : 6,
        'measure_n' : 6,
    }


    sr = simRunner()
    run_consts = pend_const, sim_const, ctrl_const
    results = sr.run_once(run_consts)
    print(results.columns)
    labels = ['x','xd','theta','thetad']        
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
    make_single_run_figure(data, show=True)

    viz = Visualizer(data, Pendulum(M=pend_const['M'], m=pend_const['m'], l=pend_const['l'], g=pend_const['g']), frameskip=20)
    viz.display_viz()

    '''
    sr = simRunner()
    params = {
        # pendulum
        'pend_rand': True,
        'M' : 5,
        'm' : 3,
        'l' : 3,
        'M_low' : 1,
        'M_high': 10,
        'm_low' : 1,
        'm_high': 8,
        'l_low' : 1,
        'l_high': 8,
        # sim
        'simtime' : 10,
        'force' : lambda t: 1/(0.5*np.sqrt(np.pi))*np.exp(-((t-1.5)/0.5)**2),
        # control
        'window' : 5,
        'measure_n': 4
    }
    results = sr.run_many(1, params, 1)
    labels = ['x','xd','theta','thetad']        
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
        ],
        axis=1,
        keys = ['ldiff', 'nldiff', 'ldiff_n', 'nldiff_n', 'state', 'mu', 'sigma', 'window']
    )

    make_single_run_figure(data)


    varnames = []
    fig3, axs3 = plt.subplots(nrows=len(data.groupby(level=0)), ncols=len(data.groupby(level=1, axis=1)), sharex=True)  
    # group by run; one row for each run.
    for i, (_, dfi) in enumerate(data.iloc[::10].groupby(level=0)):
        dfi = dfi.droplevel(0)[['state', 'nldiff', 'nldiff_n']]
        # group by col; one col for each state var
        for j, (xj, dfj) in enumerate(dfi.groupby(level=1, axis=1)):
            dfj = dfj.droplevel(1, axis=1)
            for k in dfj.columns:
                if k == 'state':
                    varnames.append(xj)
                    state_axis = axs3[i,j].twinx()
                    state_axis.set_frame_on(True)
                    state_axis.patch.set_visible(False)
                    state_axis.grid(False)
                    state_axis.get_yaxis().set_visible(False)
                    state_axis.plot(dfj[k], 'k:', label=str(k))
                else:
                    axs3[i, j].plot(dfj[k], label=str(k))
            axs3[i, j].legend()
    for j, vname in zip(axs3[0,:], varnames):
        j.set_title(vname)
    
    idx = pd.IndexSlice
    fig1, axs1 = plt.subplots(nrows=1, ncols=4)
    fig2, axs2 = plt.subplots(nrows=1, ncols=4)
    for label, ax in zip(labels, axs1.flat):
        long_data = data.droplevel(0).loc[:, idx[['ldiff','nldiff'],label]].stack().reset_index(level=1)
        long_data = long_data.melt(id_vars=['level_1'], value_vars=['ldiff', 'nldiff'], var_name='Linearity', value_name='Value')
        sns.boxplot(data=long_data, x='level_1', y='Value', hue='Linearity', ax=ax, showfliers=False, width=0.8)
        ax.set_ylabel('Value')
        ax.set_xlabel('State Var')

    for label, ax in zip(labels, axs2.flat):
        long_data = data.droplevel(0).loc[:, idx[['ldiff_n', 'nldiff_n'], label]].stack().reset_index(level=1)
        long_data = long_data.melt(id_vars=['level_1'], value_vars=['ldiff_n', 'nldiff_n'], var_name='Linearity', value_name='Value')
        sns.boxplot(data=long_data, x='level_1', y='Value', hue='Linearity', ax=ax, showfliers=False, width=0.8)
        ax.set_ylabel('Value')
        ax.set_xlabel('State Var')
    plt.show()
    '''