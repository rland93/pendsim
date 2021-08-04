---
title: 'pendsim: Developing, Simulating, and Visualizing Feedback Controlled Inverted Pendulum Dynamics'
tags:
  - Python
  - control theory
  - dynamics
  - control
  - engineering simulation
authors:
  - name: Mike Sutherland
    orcid: 0000-0001-5394-2737
    affiliation: 1
  - name: David A. Copp
    orcid: 0000-0002-5206-5223
    affiliation: 1
affiliations:
 - name: Henry Samueli School of Engineering, University of California, Irvine
   index: 1
date: 04 August 2021
---

Summary
-------

This package is a companion tool for exploring dynamics, control, and machine learning for the canonical cart-and-pendulum system. It includes a software simulation of the cart-and-pendulum system, a visualizer tool to create animations of the simulated system, and sample implementations for controllers and state estimators. The package, written in Python, can be used on any platform, including in the browser. It gives the user a plug-and-play sandbox to design and analyze controllers with this example, and is compatible with Python's rich landscape of third-party scientific programming and machine learning libraries.

Statement of need
=================

The evolution of dynamical systems can be difficult for students to visualize. Physical laboratory setups are expensive, time-consuming, and can only be used by a handful of students at a time. Virtual experiments have none of these downsides and can be used to augment course content, even for remote-only instruction. The ease of the virtual platform allows students to easily share their work, run experiments collaboratively or individually, and develop controllers or investigate system dynamics in a fast design-test loop. Instructors can use any tools available in the rich Python package ecosystem to design experiments tailored to their needs. Python allows the software to be used on any platform, including in the web browser. Powerful visualization tools (in the `matplotlib` python package) can be used to measure and record any part of the system.

These attributes make pendsim a capable companion to any control or dynamical systems course material, in either a virtual or in-person context. 


Example Usage
-------------

The software is a virtual laboratory. Users create an experiment by specifying a set of parameters: the pendulum (mass, length, friction, and so on), the simulation parameters (length of time, external forces, timestep). A controller policy designed by the user can then be applied to the system in the simulation. Finally, the user can view the results of the simulation. The ability to rapidly create and run experiments allows for fast design-prototype-test feedback loops.

An example is shown:

```python

# define simulation parameters
dt, t_final = 0.01, 5.0

def forcing_func(t):
    return 10 * np.exp( -(t/0.2)**2 )

# define pendulum parameters
pend = sim.Pendulum(
    2.0,  # Large mass, 2.0 kg
    1.0,  # Small mass, 1.0 kg
    2.0,  # length of arm, 2.0 meter
    initial_state=np.array([0.0, 0.0, 0.1, 0.0]),
)

# define controller and its parameters
kp, ki, kd = 50.0, 0.0, 5.0
cont = controller.PID(
    (kp, ki, kd)
)

# create simulation object
simu = sim.Simulation(dt, t_final, forcing_func)

# run simulation with controller and pendulum objects
results = simu.simulate(pend, cont)

# create an animation of the results
visu = viz.Visualizer(results, pend)
ani = visu.animate()
```

Package Features
================


A core pendulum/simulation module. ('sim.py')
---------------------------------------------

This simulates the system dynamics and allows for external forces on the pendulum. Users can specify:

-   pendulum parameters (e.g., masses, length, friction, etc.)

-   a time period over which to simulate

-   an external forcing function (e.g., push)

-   noise characteristics

-   a timestep for the simulation

-   a controller to use, if any

Controllers ('controller.py')
-----------------------------

Several controllers, implemented in python. These include:

-   Bang Bang controller

-   PID controller

-   LQR controller

-   MPC implementations (with package 'cvxpy')

Additionally, any control policy can be implemented by the user, by creating a new class. This allows for open-ended controller design. Controllers can dump data into the simulation results, so that intermediate control policy steps are accessible to the final results of the simulation.

Visualization ('viz.py')
------------------------

Finally, the results of a simulation can be visualized. The 'matplotlib' backend is used to draw an animation of the pendulum and any plots from the simulation. The visualization uses the results of the simulation and the pendulum to draw the pendulum, including the external and control forces applied to it. The animation module allows for the system to plot real-time simulation data (e.g., data used by the controller) side by side with the animation.

An example still from the animation can be seen here:

![Animation Still](still.png)