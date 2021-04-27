# Inverse Pendulum Simulation
A simple inverse-pendulum simulator.

[Package Documentation](http://rland93.github.io/pendulum/)

It includes implementations for controllers:
+ PID
+ Bang Bang
+ LQR
+ MPC

As well as options for custom implementations.

## Example: 
```python
# imports
from pendulum.pendulum import Pendulum
from pendulum.sim import Simulation
from pendulum.viz import Visualizer
from pendulum.controller import LQR
import numpy as np

# set 1/100th of a second timestep for the simulation
dt = .01

# create a pendulum with base mass 1.0, ball mass 0.4, and length mass 2
# at initial state x=0, xdot=0, theta=0.3, thetadot=0
pend = Pendulum(1.0, 0.4, 2.0, initial_state=np.array([0,0,0.3,0]))
# create a finite horizon LQR controller with 5 step horizon
cont = LQR(pend, dt,5)

# create a force function
alpha = 0.07 # dirac delta function frequency
beta = 20 # force magnitude
force = lambda t: beta * -1/(alpha*np.pi) * np.exp(- ( (t-1.5) / alpha)**2)

# simulate
sim = Simulation(dt, 5, force)
results = sim.simulate(pend, cont, plot=False)

# visualize
viz = Visualizer(results, pend, frameskip=1, draw_ghost=False, save=False)
viz.display_viz()
```

![Example](examplevideo.mp4)

It uses rk45 to simulate a dynamic model of the simple inverse pendulum on a cart: [Inverted Pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum).