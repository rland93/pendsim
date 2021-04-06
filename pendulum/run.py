from pendulum import Pendulum, Simulation, make_single_run_figure
import viz
from controller import MPCWithGPR, MPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import random


init = np.array([0,0,-0.1,0])
dt = 0.01
force = lambda t: 20 * 1/np.sqrt(np.pi)/0.1*np.exp(-((t-0.4)/0.1)**2.0)


sim = Simulation(dt, 7, force, solve_args={'method' : 'RK45', 'dense_output' : True})
pends, controllers = [], []
for _ in range(2):
    m = random.uniform(3,5)
    M = random.uniform(4,8)
    l = random.uniform(2,5)
    p = Pendulum(M, m, l)
    c = MPC(p, dt)
    pends.append(p)
    controllers.append(c)
results = sim.simulate(pends[0], controllers[0])
viz = viz.Visualizer(results, pends[0])
viz.display_viz()
print(results)