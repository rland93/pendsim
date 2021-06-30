from pendulum import controller, pendulum, sim, utils, viz
import numpy as np
import matplotlib.pyplot as plt
dt = 0.01
t_final = 3
pends, conts = [],[]
for i in range(10):
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
    10e-3)
  pends.append(pend)
  conts.append(cont)

c1, c2, c3, c4 = 5, 3.0, 4, .2
fshift = 2
force_fn = lambda t: c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
force_fn = lambda t: 0
if False:
  fig1, ax1 = plt.subplots()
  fx = np.linspace(0,t_final, 500)
  fy = force_fn(fx)
  ax1.plot(fx, fy)
  plt.show()

noise_scale = [1,1,1,1]
# noise_scale = [0.01, 0.01, 0.01, 0.1]
simu = sim.Simulation(
  dt, 
  t_final, 
  force_fn,
  noise_scale= noise_scale)

results = simu.simulate_multiple(pends, conts)

fig2, ax2 = plt.subplots(nrows=6)
for i, s in enumerate(results['state']):
  ax2[i].scatter(
    results.index, 
    results[('measured state', s)].values, 
    label = 'measured ' + s,
    marker = '+',
    s=5
  )
  ax2[i].plot(
    results[('est', s)], 
    label = 'estimated ' + s, 
  )
  ax2[i].scatter(
    results.index,
    results[('state', s)],
    label = 'true ' + s,
    marker= '4',
    s=10,
  )
  ax2[i].legend()

ax2[4].plot(
  results[('control action')]
)
ax2[4].plot(
  results[('forces')]
)

for i, (s, m) in enumerate(zip(results['state'], ['.', '^', '+', '_'])):
  ax2[5].scatter(
    results.index,
    np.abs((results[('state', s)].values - results[('est', s)].values)),
    label = 'error ' + s,
    marker=m
  )
ax2[5].legend()

print(np.abs((results['state'].values - results['est'].values)).sum(axis=0))

# viz = viz.Visualizer(results, pend, speed=3)
# anim = viz.animate()
# plt.show()