from pendulum import controller, pendulum, sim, utils, viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
'''
dt = 0.01
t_final = 15
c1, c2, c3, c4 = 5, 3.0, 4, .2
fshift = 2
force_fn = lambda t: c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
noise_scale = [.5, .5, .5, .5]
simu = sim.Simulation(
  dt, 
  t_final, 
  force_fn,
  noise_scale= noise_scale)

def gen_pend_cont(smooth):
  pend = pendulum.Pendulum(
    2.0,
    1.0,
    3.0,
    initial_state=np.array([0,0,0.1,0])
  )
  cont = controller.LQR_UKF(
    pend, 
    dt, 
    9, 
    [0,0,100,0], 
    10e-3,
    smooth)
  return pend, cont

smooths = [i+1 for i in range(11)]
mults = {}
for smooth in smooths:
  pends, conts = [],[]
  n_runs = 8
  for i in range(n_runs):
    pend, cont = gen_pend_cont(smooth)
    pends.append(pend)
    conts.append(cont)
  mults[smooth] = simu.simulate_multiple(pends, conts, parallel=True)
results = pd.concat(mults.values(), keys=mults.keys())
results.to_parquet('./runs2.gzip',compression='gzip')

'''

results = pd.read_parquet('./runs2.gzip')

results1 = results.loc[9.0,:].loc[0,:]

fig2, ax2 = plt.subplots(nrows=5, sharex=True)
for i, s in enumerate(results['state']):
  ax2[i].scatter(
    results1.index, 
    results1[('measured state', s)].values, 
    label = 'measured ' + s,
    marker = '+',
    s=5
  )
  ax2[i].plot(
    results1[('est', s)], 
    label = 'estimated ' + s, 
  )
  ax2[i].scatter(
    results1.index,
    results1[('state', s)],
    label = 'true ' + s,
    marker= '4',
    s=10,
  )
  ax2[i].legend()

ax2[4].plot(
  results1[('control action')]
)
ax2[4].plot(
  results1[('forces')]
)
fig3, ax3 = plt.subplots(nrows=4)
for i, (s, m) in enumerate(zip(results1['state'], ['.', '^', '+', '_'])):
  results1[('est diff',s)] = results1[('est', s)] - results1[('measured state', s)]
  results1[('est err',s)] = (results1[('state', s)] - results1[('est',s)]).abs()
  ax3[i].plot(
    results1[('est diff',s)],
    label = 'est diff ' + s,
    marker=m
  )
  ax3[i].plot(
    results1[('est err',s)],
    label = 'est err ' + s,
    marker=m
  )
  ax3[i].legend()



for s in results['state']:
  results[('meas err',s)] = (results[('state',s)] - results[('measured state',s)]).abs()
  results[('ukf err',s)] = (results[('state',s)] - results[('est',s)]).abs()

results = results.unstack(level=0)

results2 = pd.melt(
  results[
    ['ukf err', 'meas err']
  ],
  var_name=['var', 'x_i', 'smooth'],
  value_name='val'
  )

results2['smooth'] = results2['smooth'].astype('category')


print(results2)

g = sns.FacetGrid(results2, col='x_i')
g.map_dataframe(sns.boxplot, data=results2, x='val', y='smooth', hue='var', showfliers=False)
plt.show()