import sys
sys.path[0] += '/../'
print(sys.path)
from pendulum import controller, pendulum, sim, utils, viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dt, t_final = 0.01, 6
c1, c2, c3, c4, fshift = 5, 3.0, 4, .2, 3
def force_fn(t):
  return c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
simu = sim.Simulation(dt, t_final, force_fn, noise_scale=[.5, .5, .5, .5])
pend = pendulum.Pendulum(4,4,3.0, cfric=.5, pfric=0.3, initial_state=np.array([0,0,0.1,0]))
cont = controller.LQR_UKF(pend, dt, 9, [0,0,100,0], 10e-3, 10)
results = simu.simulate(pend, cont)

fig3, ax3 = plt.subplots()
print(results.columns)
for e in results['energy']:
  ax3.plot(results[('energy',e)], label=e)
ax3.legend()
visu = viz.Visualizer(results, pend, speed=6)
anim = visu.animate()

fig1, ax1 = plt.subplots(nrows=5, sharex=True)
for i, s in enumerate(results['state']):
  ax1[i].scatter(results.index, results[('measured state', s)].values, label = 'measured ' + s,marker = '+',s=5)
  ax1[i].plot(results[('est', s)], label = 'estimated ' + s, )
  ax1[i].scatter(results.index,results[('state', s)],label = 'true ' + s,marker= '4', s=10)
  ax1[i].legend()
ax1[4].plot(results[('control action')])
ax1[4].plot(results[('forces')])

fig2, ax2 = plt.subplots()
for s in results['state']:
  results[('measured error',s)] = (results[('measured state',s)] - results[('state',s)]).abs()
  results[('estimate error',s)] = (results[('est',s)] - results[('state',s)]).abs()
boxresults = results[['measured error', 'estimate error']].melt(var_name=['measurement', 'xi'])
print(boxresults)
sns.violinplot(data=boxresults, x='value', y='measurement', ax=ax2)
plt.show()
