import numpy as np
import pandas as pd
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

    def calculate_xdot(self, x, u):
        '''
        Calculate xdot = f(x, u) where x is the state vector 
        xdot is a vector [xdot, xddot, tdot, tddot] It's may
        be a little bit confusing because xdot indicates either
        the state vector or the cart absolute position/velo
        depending on context.
        '''
        # state =   [x, xdot, theta, thetadot]
        sin_t = np.sin(x[2])
        cos_t = np.cos(x[2])
        # A^-1     2x2
        A = np.linalg.inv(
            np.array([
            [self.M + self.m,       -self.m * self.l * cos_t],
            [-cos_t,                self.l]
            ], dtype='float')
        )
        # B  2 @ 1
        B = np.array([
            [-self.m*self.l*x[3]*x[3] * sin_t + u],
            [self.g * sin_t]
        ])
        solution = A @ B

        xd = x[1]
        xdd = solution[0][0] - x[1] * self.cfric
        td = x[3]
        tdd = solution[1][0] - x[3] * self.pfric

        return np.array([xd, xdd, td, tdd])

    def update_rk4(self, x, u, dt):
        '''
        Update the pendulum state using the rk4 method
        '''
        k1 = self.calculate_xdot(x, u) * dt
        k2_state = x + k1 * 0.5 * dt
        k2 = self.calculate_xdot(k2_state, u)
        k3_state = x + k2 * 0.5 * dt
        k3 = self.calculate_xdot(k3_state, u)
        k4_state = x + k3 * dt
        k4 = self.calculate_xdot(k4_state, u)
        state = x + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt
        # wrap pi
        # state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        return state

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
    def __init__(self, pend, dt, t_final, force, control_every, noise_scale):
        self.pend = pend # pendulum to be simulated
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.force = force # forcing function
        self.control_every = control_every # control action interval, 1 = control every dt, 2 = control every other dt, etc.
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

            # controller takes action every `control_every` steps
            if n % self.control_every == 0:
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

            # update state, time
            x_k = self.pend.update_rk4(x_k, u_k, self.dt)
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
            sim_const['every'],
            sim_const['noise']
        )
        ctrl = controller.MPCWithGPR(
            pendulum,
            sim_const['dt'],
            ctrl_const['measure_n'],
            ctrl_const['window'],
            sim_const['every']
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
                'every' : 10,
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
                'every' : 10,
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
    # get view of data from data as nparray
    x = data.index
    print(x)
    '''
    get_view = lambda key: np.stack(data[key].values)
    labels = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    x = data.index
    for i in range(get_view('state').shape[1]):
        # make plots
        fig = plt.figure('pend_nl_'+str(i), figsize=(7.5, 10), tight_layout=True, facecolor='silver')
        state = fig.add_subplot(411)
        forces = fig.add_subplot(412, sharex=state)
        pred = fig.add_subplot(413, sharex=state)
        errors = fig.add_subplot(414, sharex=state)
        # grids
        state.grid(True,    which='both', color='lightgrey')
        errors.grid(True,   which='both', color='lightgrey')
        pred.grid(True,     which='both', color='lightgrey')
        forces.grid(True,   which='both', color='lightgrey')
        # titles
        state.set_title('State: ' + labels[i])
        forces.set_title('Forces')
        pred.set_title(r'Predicted Divergence from Nominal Model ($\mu$)')
        errors.set_title('Prediction Error')
        # axis labels
        state.set_ylabel(labels[i])
        state.set_xlabel('time (s)')
        forces.set_ylabel('Force (N)')
        forces.set_xlabel('time (s)')
        pred.set_ylabel(r'$\mu$')
        pred.set_xlabel('time (s)')
        errors.set_ylabel('Pred. error')
        errors.set_xlabel('time (s)')
        # data
        state.plot(x, get_view('state')[:,i],           label=labels[i])
        state.fill_between(x, get_view('state')[:,i],   alpha=0.06, facecolor='k')
        forces.plot(x, get_view('forces'),              label='extern. force')
        forces.plot(x, get_view('control action'),      label='actuation', linestyle='--')
        pred.fill_between(x, get_view('upper_conf')[:,i], get_view('lower_conf')[:,i], alpha=0.2, facecolor='k', label=r'$\pm\sigma$')
        pred.plot(x, get_view('mu')[:,i],               label=('divergence (' + labels[i] + ') predicted by GP'))
        errors.plot(x, get_view('linear_error')[:,i],   label=('nominal (' + labels[i] + ')'), linestyle='--')
        errors.plot(x, get_view('nonlinear_error')[:,i],label=('nominal + GP (' + labels[i] + ')'), linestyle='-')
        errors.fill_between(x, get_view('sigma')[:,i], -1 * get_view('sigma')[:,i],  facecolor='k', label='uncertainty', alpha=0.2)

        # legends
        state.legend(framealpha=1, facecolor='inherit', loc='best')
        errors.legend(framealpha=1, facecolor='inherit', loc='best')
        pred.legend(framealpha=1, facecolor='inherit', loc='best')
        forces.legend(framealpha=1, facecolor='inherit', loc='best')
        if save:
            fig.savefig(path_prefix + 'unmodeled_dyn_pend_nl_'+str(i)+'.png', dpi=200, facecolor='gainsboro')
    if show:
        plt.show()
    # close out
    plt.close(fig)
    '''




if __name__ == '__main__':
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
    results = sr.run_many(64, params, 1)
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