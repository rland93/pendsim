import viz
from scipy.signal import cont2discrete
import cvxpy as cp
import numpy as np
from collections import defaultdict
import pandas as pd
import scipy.integrate
import random
from matplotlib import pyplot as plt
import math

class Pendulum(object):
    def __init__(self, M, m, l, g=9.81, cfric=0.04, initial_state=np.array([0,0,0,0])):
        self.M = M
        self.m = m 
        self.l = l
        self.g = g
        self.cfric = cfric
        self.y_0 = initial_state

    def system_dynamics(self, t, y):
        xdd = (y[4] + self.m * self.g * np.sin(y[2]) * np.cos(y[2]) - self.m * self.l * y[3] * y[3] *  np.sin(y[2])) / (self.M + self.m - self.m * np.cos(y[2]) * np.cos(y[2]))
        tdd = (xdd * np.cos(y[2]) + self.g * np.sin(y[2]) ) / self.l
        return np.array([y[1] * (1 - self.cfric), xdd, y[3], tdd, y[4]])

    def solve(self, dt, system_state, u, solve_args):
        # roll state and u into a single array
        y = np.empty( (5) )
        y[:4] = system_state
        y[4] = u
        # solve system
        sol = scipy.integrate.solve_ivp(self.system_dynamics, (0, dt), y, **solve_args)
        return sol.y[:4, -1], sol.y[4, -1]

    def get_energy(self, y):
        # cart vel^2
        v_c2 = y[1] * y[1]
        # pend vel^2
        v_p2 = v_c2 - 2 * self.l * y[1] * y[3] * np.cos(y[2]) + self.l * self.l * y[3] * y[3]
        ke = 0.5 * self.M * v_c2 + 0.5 * self.m * v_p2
        pe = self.m * self.g * self.l * np.cos(y[2])
        return ke, pe, ke + pe

class MPC(object):
    def __init__(self, pend, dt, window):
        self.T = window
        self.u_max = 100
        A = np.array([
            [0, 1.0, 0, 0],
            [0, 0, -pend.m * pend.g / pend.M, 0.0],
            [0, 0, 0, 1.0],
            [0, 0, pend.g * (pend.m * pend.M) / (pend.l * pend.M), 0.0]
        ])
        B = np.array([
            [0.0],
            [1.0 / pend.M],
            [0.0],
            [-1.0 / (pend.l * pend.M)]
        ])
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
        problem.solve('ECOS', max_iters=50, verbose=True)


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
    
    def simulate(self, pendulum, controller):
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
            state, _ = pendulum.solve(self.dt, state, force, {})
            for k, v in data.items():
                datas[k].append(v)
        plt.ioff()
        return pd.DataFrame(datas, index=times)

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
    results = sim.simulate(pend, cont)
    viz = viz.Visualizer(results, pend, frameskip=1, draw_ghost=True)
    viz.display_viz()