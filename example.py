from pendulum import sim, controller, pendulum, viz
import numpy as np
from pendulum.lqr_gpr2 import LQR_GPR2
import pandas as pd
import matplotlib.pyplot as plt
import skopt

dt = .01
pend = pendulum.Pendulum(
    4.0,
    3,
    2.0,
    initial_state=np.array([0,0,-np.pi,-0.2])
)


fa = 2.0
fshift = 6.0
c1,c2,c3,c4 = 5.0, 8, 50, 0.5
'''
dirac = lambda t: -fm*1/(fa*np.sqrt(np.pi)) * np.exp(-((t-fshift)/fa)**2)
sine = lambda t: fm * np.sin(2*t)
zero = lambda t: 0
'''
diracsine = lambda t: c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
sim = sim.Simulation(
        dt,
        8,
        diracsine)

Q = [1,1,1000,1,0.003]
cont = LQR_GPR2(pend, dt, 8, 10, Q)
results = sim.simulate(pend, cont, plot=False)

for k2 in ['x','xd','t','td']:
    results[('lerr',k2)] = results[('lpred',k2)].shift(1, fill_value=0.0) - results[('state',k2)]
    results[('nlerr',k2)] = results[('nlpred',k2)].shift(1, fill_value=0.0) - results[('state',k2)]
    results[('lerrabs', k2)] = results[('lerr', k2)].abs()
    results[('nlerrabs', k2)] = results[('nlerr', k2)].abs()
    results[('muabs', k2)] = results[('mu', k2)].abs()



fig1, axs1 = plt.subplots(nrows=4, tight_layout=True, sharex=True)
fig2, axs2 = plt.subplots(nrows=2, tight_layout=True, sharex=True)
fig3, axs3 = plt.subplots(nrows=4, tight_layout=True, sharex=True)

for i, k2 in enumerate(['x','xd','t','td']):
    axs1[i].set_title(k2)
    axs1[i].axhline(0, label='reference')
    axs1[i].plot(results[('lerrabs',k2)],'k--', label='lerr')
    axs1[i].plot(results[('nlerrabs',k2)],'r', label='nlerr')
    axs1[i].legend()
    axs2[1].plot(results[('state wrapped',k2)], label=k2)
    axs2[1].legend()
    axs3[i].plot(results[('mu', k2)], 'r--', label='mu t')
    axs3[i].plot(results[('train', k2)], 'k', label='train t')
    axs3[i].legend()
    
axs2[0].plot(results[('control action','control action')], 'b', label='action')
axs2[0].plot(results[('forces','forces')], 'r', label='force')
axs2[0].legend()

viz = viz.Visualizer(results, pend, speed=1, plotpoints=40)

ghosts = {
    'nlpred' : ('green', ':'),
    'lpred' : ('blue', ':')
}

plots = {
    ('train','t') : {
        'marker' : '.',
        'color' : 'k',
        'label' : 'linear error'
    },
    ('mu','t') : {
        'marker' : ',',
        'color' : 'r',
        'label' : 'nonlinear error'
    },
}
viz.display_viz(plots)