from pendulum import Pendulum, Simulation, make_single_run_figure
from controller import MPCWithGPR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import random


init = np.array([0,0,-0.1,0])
dt = 0.1
force = lambda t: np.pi/2 - np.arctan( 10 * (t - 1)**2.0 )



sim = Simulation(dt, 10, force, solve_args={'method' : 'RK45', 'dense_output' : True})
pends, controllers = [], []
for _ in range(2):
    m = random.uniform(3,5)
    M = random.uniform(4,8)
    l = random.uniform(2,5)
    p = Pendulum(M, m, l)
    c = MPCWithGPR(p, dt)
    pends.append(p)
    controllers.append(c)
results = sim.simulate_many(pends, controllers)
for run in results.groupby(level=0):
    print(run[1].droplevel(0).columns)
    make_single_run_figure(run[1].droplevel(0))
plt.show()



'''
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
    'simtime' : 0.3,
    'force' : lambda t: -c * 1/abs(a*np.pi) * np.exp( -((t-b)/a)**2 ),
    'noise' : 0
}
ctrl_const = {
    'window' : 6,
    'measure_n' : 6,
}


sr = pendulum.simRunner()
run_consts = pend_const, sim_const, ctrl_const
results = sr.run_once(run_consts)
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

pendulum.make_single_run_figure(data)
viz = pendulum.Visualizer(data, pendulum.Pendulum(M=pend_const['M'], m=pend_const['m'], l=pend_const['l'], g=pend_const['g']), frameskip=20)
viz.display_viz()
'''