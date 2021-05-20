from pendulum import sim, controller, pendulum, viz
import numpy as np
from pendulum.lqr_gpr import LQR_GPR
import pandas as pd
import matplotlib.pyplot as plt
import skopt

dt = .005
pend = pendulum.Pendulum(
    4.0,
    3,
    2.0,
    initial_state=np.array([0,0,0.5,-0.2])
)


fa = 0.3
fshift = 6.0
fm = 40
fm2= 20
dirac = lambda t: -fm*1/(fa*np.sqrt(np.pi)) * np.exp(-((t-fshift)/fa)**2)
sine = lambda t: fm * np.sin(2*t)
zero = lambda t: 0

diracsine = lambda t: fm * np.sin(2*t) * fm2/(fa*np.sqrt(np.pi)) * np.exp(-((t-fshift)/fa)**2)
sim = sim.Simulation(
        dt,
        10,
        dirac)


Q = [1,1,1000,1,0.003]
cont = LQR_GPR(pend, dt, 10, 5, Q)
results = sim.simulate(pend, cont, plot=False)

n = 8

for k2 in ['x','xd','t','td']:
    results[('lerr',k2)]=results[('lpred',k2)].shift(n, fill_value=0.0) - results[('state',k2)]
    results[('nlerr',k2)]=results[('nlpred',k2)].shift(n, fill_value=0.0) - results[('state',k2)]
    results[('lpred', k2)] = results[('lpred', k2)].shift(-n, fill_value=0.0)
    results[('nlpred', k2)] = results[('nlpred', k2)].shift(-n, fill_value=0.0)
    results[('lerrabs', k2)] = results[('lerr', k2)].abs()
    results[('nlerrabs', k2)] = results[('nlerr', k2)].abs()

fig1, axs1 = plt.subplots(nrows=4, tight_layout=True, sharex=True)
for i, k2 in enumerate(['x','xd','t','td']):
    axs1[i].set_title(k2)
    axs1[i].axhline(0, label='reference')
    axs1[i].plot(results[('lerrabs',k2)],'k--', label='lerr')
    axs1[i].plot(results[('nlerrabs',k2)],'r', label='nlerr')
    axs1[i].legend()
    
fig2, axs2 = plt.subplots(nrows=2, tight_layout=True)
axs2[0].plot(results[('control action','control action')], 'b', label='action')
axs2[0].plot(results[('forces','forces')], 'r', label='force')
axs2[0].legend()

print(results.columns)


viz = viz.Visualizer(results, pend, speed=4)
viz.display_viz(ghost1='nlpred', ghost2='lpred')