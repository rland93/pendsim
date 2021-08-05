"""
This is intended to be a full demonstration of the capabilities
of this package. We define a pendulum and simulation, with an
external impulse force. We use an Unscented Kalman Filter with 
PID control policy for the experiment.

We visualize the results with a chart, as well as an animation
of the pendulum alongside a chart with the measured state, the
estimate produced by the controller, and the true state.
"""

from pendsim import sim, controller, viz
import numpy as np
import matplotlib.pyplot as plt

# define simulation parameters
dt, t_final = 0.005, 8.0

# external force is an impulse function
def extforce(t):
    return 20 * np.exp(-(((t - 0.25) / 0.1) ** 2))


simu = sim.Simulation(
    dt, t_final, extforce, noise_scale=np.array([0.02, 0.02, 0.02, 0.02])
)

# define a pendulum
pend = sim.Pendulum(2.0, 1.0, 3.0)

# make a PID controller
kp, ki, kd = 50.0, 0.0, 2.0
cont = controller.PID_UKF((kp, ki, kd), pend, dt, var_t=0.1)

# run the simulation
results = simu.simulate(pend, cont)
# plot some results
fig1, (ax1, ax2) = plt.subplots(nrows=2)

# plot energies
ax1.plot(results[("energy", "kinetic")], "r--", label="Kinetic Energy")
ax1.plot(results[("energy", "potential")], "b:", label="Potential Energy")
ax1.plot(results[("energy", "total")], "g-", label="Potential Energy")
ax1.plot(results[("forces", "forces")], "k-", label="External Force")
ax1.legend()

# plot some states
ax2.plot(results[("state", "t")], "k-", label=r"$\theta$")
ax2.plot(results[("state", "xd")], "b-", label=r"$\dot{x}$")
ax2.legend()

# create a visualizer
visu = viz.Visualizer(results, pend, dt)

plotdata = {
    ("measured state", "t"): {
        "type": "scatter",
        "plotpoints": 100,
        "color": "grey",
        "label": r"$\theta$" + " (Measured)",
    },
    ("state", "t"): {
        "type": "line",
        "linestyle": "-",
        "plotpoints": 200,
        "color": "black",
        "label": r"$\theta$" + " (True)",
    },
    ("est", "t"): {
        "type": "line",
        "linestyle": "--",
        "plotpoints": 200,
        "color": "red",
        "label": r"$\theta$" + " (Estimated)",
    },
}

anim = visu.animate(pltdata=plotdata)

plt.show()
