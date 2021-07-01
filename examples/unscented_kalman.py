from re import A
import sys
sys.path[0] += '/../'
from pendulum import controller, pendulum, sim, utils, viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dt, t_final = 0.01, 5
c1, c2, c3, c4, fshift = 60, 2.0, 4, .2, 3
def force_fn(t):
  return c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
noise_scale = [.1,.1,.1,.1]
simu = sim.Simulation(dt, t_final, force_fn, noise_scale=noise_scale)
pend = pendulum.Pendulum(4,1,3.0, cfric=.5, pfric=0.3, initial_state=np.array([0,0,0.1,0]))

Q = [0,0,1,0]
R = 5e-6

cont1 = controller.LQR(pend, dt, 24, Q, R)
cont2 = controller.LQR_UKF(pend, dt, 24, Q, R, 6)

res_lqr = simu.simulate(pend, cont1)
res_ukf = simu.simulate(pend, cont2)

fig1, ax1 = plt.subplots(nrows =2, figsize=(6,4), tight_layout=True)
action_df = pd.concat((res_lqr[('control action', 'control action')].abs(), res_ukf[('control action', 'control action')].abs()), axis=1, keys=['LQR only','LQR + state est'])
action_df = action_df.melt(var_name='type', value_name='control action')
sns.boxplot(data=action_df,x='control action', y='type', ax=ax1[0])
sns.stripplot(data=action_df,x='control action',y='type',linewidth=0,size=3,color='.3',ax=ax1[0])

for s in res_ukf['state']:
  res_ukf[('est err', s)] = (res_ukf[('est', s)] - res_ukf[('state', s)]).abs()
  res_ukf[('meas err', s)]= (res_ukf[('measured state', s)] - res_ukf[('state', s)]).abs()

err_df = res_ukf[['meas err', 'est err']].melt(var_name=['measure', 'x_i'])
sns.boxplot(data=err_df,x='value',y='x_i',hue='measure',ax=ax1[1])

fig2, ax2 = plt.subplots(nrows=6, sharex=True, tight_layout=True)
for i, s in enumerate(res_ukf['state']):
  ax2[i].scatter(res_ukf.index, res_ukf[('measured state', s)].values, label = 'measured ' + s,marker = '+',s=5)
  ax2[i].plot(res_ukf[('est', s)] ,  label = 'estimated ' + s)
  ax2[i].plot(res_ukf[('state', s)], label = 'true ' + s)
  ax2[i].legend()
  ax2[5].plot(res_ukf[('var',s)], label='std '+s)
ax2[4].plot(res_ukf[('control action')], 'b')
ax2[4].plot(res_ukf[('forces')], 'r')
ax2[5].legend()

visu1 = viz.Visualizer(res_ukf, pend, speed=4)
anim1 = visu1.animate()

visu2 = viz.Visualizer(res_lqr, pend, speed=4)
anim2 = visu2.animate()
plt.show()