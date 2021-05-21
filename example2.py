from pendulum import sim, controller, pendulum, viz
import numpy as np
from pendulum.lqr_gpr2 import LQR_GPR2
import pandas as pd
import matplotlib
pgf = False
if pgf:
    matplotlib.use('pgf')
import matplotlib.pyplot as plt
import skopt

font = {'size'   : 8 }
matplotlib.rc('font', **font)


dt = .01
pend = pendulum.Pendulum(
    4.0, 3, 2.0, initial_state=np.array([0,0,-np.pi+0.1,-0.2]))
fa = 2.0
fshift = 4
c1,c2,c3,c4 = 5.0, 8, 50, 0.5
diracsine = lambda t: c1 * np.sin(c2*t) \
    * c3/(c4*np.sqrt(np.pi)) \
        * np.exp(-((t-fshift)/c4)**2)
sim = sim.Simulation(dt, 10, diracsine)
Q = [1,1,1000,1,0.0035]
cont = LQR_GPR2(pend, dt, 8, 12, Q)
results = sim.simulate(pend, cont, plot=False)
results[('mu','t')]
results[('sigma','t')]
results[('sigma abs','t')] = results[('sigma','t')].abs()
results[('lerr','t')] = results[('lpred','t')].shift(1, fill_value=0.0) - results[('state','t')]
results[('nlerr','t')] = results[('nlpred','t')].shift(1, fill_value=0.0) - results[('state','t')]
results[('lerrabs', 't')] = results[('lerr', 't')].abs()
results[('nlerrabs', 't')] = results[('nlerr', 't')].abs()

fig1, axs1 = plt.subplots(nrows=2, sharex=True, figsize=(5,2.8))
axs1[0].plot(results[('lpred','t')],'b--', label='nominal')
axs1[0].plot(results[('nlpred','t')],'r', label='nominal + GP')
axs1[0].plot(results[('state', 't')],'k', label='true')
axs1[0].set_ylabel('Theta')
axs1[0].legend()

axs1[1].set_ylabel('|Error| in Theta')
axs1[1].plot(results[('lerrabs','t')][.02:],'k--', label='nominal error')
axs1[1].plot(results[('nlerrabs','t')][.02:],'r', label='nominal + GP error')

fbx = results[('nlerrabs','t')][.02:].index.to_numpy(dtype=np.float64)
fbmu = results[('nlerrabs','t')][.02:].to_numpy(dtype=np.float64)
fby1 = fbmu - results[('sigma abs','t')][.02:].to_numpy(dtype=np.float64)
fby2 = fbmu + results[('sigma abs','t')][.02:].to_numpy(dtype=np.float64)

for f in [fbx, fby1, fby2]:
    print(f.shape)

axs1[1].fill_between(fbx, fby1, fby2)
axs1[1].set_title('')
axs1[1].set_xlabel('Time (shared)')
axs1[1].legend()

if not pgf:
    plots = {
        ('lerrabs','t') : {
            'marker' : '.',
            'color' : 'k',
            'label' : 'nominal error'
        },
        ('nlerrabs','t') : {
            'marker' : 'o',
            'color' : 'r',
            'label' : 'nominal + GP error'
        },
    }

    viz = viz.Visualizer(results, pend, speed=2, plotpoints=40)
    viz.display_viz(plots)
else:
    fig1.savefig('./result.pgf', format='pgf', dpi=200)