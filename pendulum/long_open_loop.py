import controller
import pendulum.Visualizer
import numpy as np

# a sequence of open loop forces
forces = [
    (50, 1, .5),
    (-100, 6, .25),
    (5, 9, 5),
    (-25, 12, 1),
    (-5, 26, 3),
    (400, 35, .1),
    (-300, 38, .12)
]
# pendulum attributes
pend_init_state = np.array([0,0,np.pi/2,0])
p = pendulum.Pendulum(
    5,
    1,
    4,
    9.81, 
    cfric=0.1, 
    pfric=0.01, 
    init_state=pend_init_state)

sim = pendulum.Simulation(p, 0.001, 50, forces, 0)
data = sim.simulate(controller.NoController())
plot = pendulum.Visualizer(
    data, 
    p, 
    frameskip=25, 
    cart_display_aspect=2,
    save=False, 
    viz_size=(20,7), 
    viz_xcenter = -10,
    viz_window_size=(16,9)
)
plot.display_viz()
plot.display_plots()
plt.show()