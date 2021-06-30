from collections import defaultdict
from pendulum.controller import Controller
import random
import numpy as np
import matplotlib.pyplot as plt
from pendulum.utils import array_to_kv
from multiprocessing.dummy import Pool
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import copy
from pendulum.utils import wrap_pi

class Simulation(object):
    '''The simulation class includes methods for simulating a pendulum(s)
    with its controller(s), (and some methods for processing pendulum data)
    '''
    def __init__(self, dt, t_final, force, noise_scale=None):
        '''New Simulation object

        Parameters
        ----------
        dt : :obj:float
            timestep in s.
        t_final : :obj:float
            run the simulation from 0s to `t_final`
        force : :obj:function
            a function which takes 1 argument, t, and returns
            a force in N based on t.
        noise_scale : :obj:array or scalar
            a scalar or array for state noise
        '''
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.force = force # forcing function
        self.noise_scale = noise_scale
    
    def simulate(self, pendulum, controller, **kwargs):
        '''Simulate a pendulum/controller combination from t_0 to t_final.

        Parameters
        ----------
        pendulum : :obj:Pendulum
            pendulum object to simulate
        controller : :obj:Controller
            controller object to simulate

        Returns
        -------
        :obj:pd.DataFrame
            The simulation data.
        '''
        # unpack kwargs
        plot = kwargs.pop('plot', False)
        times = []
        t = 0
        state = pendulum.y_0
        datas = defaultdict(list)
        statelabels = ['x', 'xd', 't', 'td']
        while t <= self.t_final:
            times.append(t)
            t += self.dt

        if plot:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter([],[])
            lplot1 = ax.plot([], [])
            lplot2 = ax.plot([], [])
            print(type(lplot1), type(lplot2))

        for k, t in tqdm(enumerate(times), total=len(times)):
            data = {}
            force = self.force(t)

            # Control policy
            data.update(array_to_kv('state', statelabels , state))
            if self.noise_scale is not None:
                noisy_state = state + np.random.normal(0, scale=self.noise_scale)
                data.update(array_to_kv('measured state', statelabels, noisy_state))
                action, controller_data = controller.policy(noisy_state, t, self.dt)
            else:
                action, controller_data = controller.policy(state, t, self.dt)
            data.update(controller_data)

            # Simulation Data
            data[('energy','kinetic')], data[('energy', 'potential')], data['energy','total'] = pendulum.get_energy(state)
            data[('forces','forces')] = force
            data[('control action','control action')] = action

            # Simulation solution
            force += action
            state, _ = pendulum.solve(self.dt, state, force)
            for k, v in data.items():
                datas[k].append(v)
        if plot:
            plt.ioff()
        return pd.DataFrame(datas, index=times)
    
    def simulate_multiple(self, pendulums, controllers, parallel=True):
        '''Simulate many pendulum/controller combinations.

        If you are using random variables to populate pendulums/controllers, 
        make sure that you deepcopy those objects, because some parameters may
        be shared internally.

        Parameters
        ----------
        pendulums : :obj:`list` of :obj:`Pendulum`
            the pendulums to simulate, in order
        controllers : :obj:`list` of :obj:`Controller`
            the controllers to simulate, in order. Must be same 
            length as pendulums.
        parallel : :obj:`bool`, optional
            whether to simulate in parallel, by default True. simulation
            is almost completely CPU bound so if you can use this, do.

        Returns
        -------
        :obj:pd.DataFrame
            MultiIndex DataFrame of simulations. Simulations are stacked on axis 0,
            so axis=0 level=0 contains the run # and axis=0 level=1 contains individual
            simulation data.

        Raises
        ------
        ValueError
            If pendulums/controllers are not equal length.
        '''
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