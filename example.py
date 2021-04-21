from pendulum.pendulum import Pendulum
from pendulum.sim import Simulation
from pendulum.viz import Visualizer
from pendulum.controller import LQR, MPC, NoController
import numpy as np

dt = .01
pend = Pendulum(
    1.0,
    0.4,
    2.0,
    initial_state=np.array([0,0,0.3,0])
)
cont = LQR(
    pend,
    dt,
    7
)
sim = Simulation(
    dt,
    10,
    lambda t: 0
)
results = sim.simulate(pend, cont, plot=False)
print(results)
viz = Visualizer(results, pend, frameskip=1, draw_ghost=False)
viz.display_viz()