import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from math import sin, cos, pi
from scipy.integrate import solve_ivp

class Pendulum(object):
    '''
    An inverted pendulum object.
    '''
    def __init__(self, M, m, l, g, init_cond=np.array([0,0,0,0])):
        '''
        Parameters
        ----------

        M: float
            The cart mass
        m: float
            The pendulum mass
        l: float
            The pendulum length
        g: float
            Acceleration due to gravity
        '''
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.E = 0 #: current energy (kinetic plus potential)
        self.xdd = 0 #: current xddot (linear acceleration cart)
        self.tdd = 0 #: current thetaddot (angular acceleration pendulum)
        self.x = init_cond 
        '''
        x: 1d numpy array of float
            The state of the pendulum, set to initial conditions. [x, xdot, theta, thetadot]
        '''
    
    def update_accel(self, u):
        '''
        Update accelerations xddot, thetaddot from current state, given an input force.

        Parameters:
        -----------
        u: float
            Force on the cart pointing right, in newtons.
        '''
        sin_t = np.sin(self.x[2])
        cos_t = np.cos(self.x[2])

        self.xdd = (u + self.m*self.g*sin_t *cos_t - self.m * self.l * self.x[3] * self.x[3] * sin_t) / (self.M+self.m-self.m*cos_t*cos_t)
        self.tdd = self.xdd * cos_t / self.l + self.g * sin_t / self.l

    def pend_update(self, dt, u):
        '''
        Parameters:
        -----------
        dt: float
            The length of the update step (seconds, I guess?)
        '''

        # update acceleration xddot_k, thddot_k = f(x_k, u_k)
        self.update_accel(u)

        # calculate new position x1 = x0 + v0 * t
        self.x[0] += self.x[1] * dt
        self.x[2] += self.x[3] * dt

        # calculate new velocity v1 = v0 + a0 * t
        self.x[1] += self.xdd * dt
        self.x[3] += self.tdd * dt

    def calculate_energy(self):
        '''
        calculate the energy of the current state

        Returns:
        --------
        float: the energy of the current state
        '''
        # cart vel. squared
        v_c2 = self.x[1] * self.x[1] 

        # pendulum velocity squared
        v_p2 = v_c2 - 2 * self.l * self.x[1] * self.x[3] * np.cos(self.x[2])\
            + self.l * self.l * self.x[3] * self.x[3] 

        # energy of current state
        self.E = 0.5 * self.M * v_c2 + .5 * self.m * v_p2 + self.m * self.g\
            * self.l * np.cos(self.x[2])

    def calculate_momentum(self):
        '''
        calculate the (x-direction) linear momentum of the current state.

        Returns:
        --------
        (float, float): the current momentums of the cart and the pendulum.
        '''
        p_cart = self.M * self.x[0] 
        p_pend = self.m * (self.x[1] + self.x[3] * np.sin(self.x[2]))
        return (p_cart, p_pend)

class Simulation(object):
    def __init__(self, pendulum, dt, t_final, u=[()]):
        '''
        Parameters:
        -----------
        pendulum: Pendulum
            The pendulum to be simulated
        dt: float
            The time step of the simulation
        t_final: float
            Simulate up to this time
        u: list of (float, float, float)
            List of force magnitude, start time, and end time pairs.
        '''
        self.pendulum = pendulum
        self.dt = dt
        self.t_final = t_final
        self.u = u
        self.u_k = 0
        self.k = 0 #: the kth simulation timestep
    
    def simulate(self):
        t = 0
        times = []
        xs = []
        xds = []
        ts = []
        tds = []
        energies = []
        ps_cart = []
        ps_pend = []
        ps_tot = []

        while t <= self.t_final:
            # record the current state, energies, etc.
            times.append(t)
            # append current state
            xs.append(self.pendulum.x[0])
            xds.append(self.pendulum.x[1])
            ts.append(self.pendulum.x[2])
            tds.append(self.pendulum.x[3])
            # append energies, momentums
            energies.append(self.pendulum.calculate_energy())
            p_cart, p_pend = self.pendulum.calculate_momentum()
            ps_cart.append(p_cart) 
            ps_pend.append(p_pend)
            ps_tot.append(p_cart+p_pend)
            self.u_k = 0
            # add up all present forces
            for i, force in enumerate(self.u):
                if force[1] > force[2]:
                    raise ValueError("Force {}: start time of force (={}) is greater than end time of force (={})".format(i, force[1], force[2]))
                if force[1] < t < force[2]:
                    self.u_k += force[0]
            # update pendulum state
            self.pendulum.pend_update(self.dt, self.u_k)
            t += self.dt

        return pd.DataFrame(
            index=times, 
            data={
                'x':xs, 
                'xdot': xds,
                'theta':ts,
                'thetadot':tds,
                'energy':energies, 
                'cart momentum': ps_cart,
                'pendulum momentum': ps_pend, 
                'total momentum': ps_tot})

pendulum = Pendulum(10,5,5,9.81, np.array([0,0,0.5,0]))
sim = Simulation(pendulum, .01, 5,[(1,0.5,2.5)])
df = sim.simulate()


C_WIDTH = 2
CART_Y = 1
CART_DISPLAY_WIDTH = 2

# Set up figure
gs = GridSpec(2,5)
fig = plt.figure(figsize=(20,8))

# Animation window
ax1 = fig.add_subplot(gs[:,:2])
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
cart = patches.Rectangle((-C_WIDTH*.5,CART_Y), width=CART_DISPLAY_WIDTH, height=-CART_Y, ec='black', fc='salmon')
mass = patches.Circle((0,0), radius=np.sqrt(5)*.14, fc='skyblue', ec='black')
line = patches.FancyArrow(0,0,1,1)
time_text = text.Annotation('', (4,4), xycoords='axes points')
ax1.grid(True)

# Initialize animation object
def init():
    ax1.add_patch(mass)
    ax1.add_patch(cart)
    ax1.add_artist(time_text)
    ax1.add_patch(line)
    return [mass, cart, time_text, line]

# Animate ith frame
def animate(i):
    x = df['x'][i] # position
    th = df['theta'][i] # angle
    massxy = (x + pendulum.l * np.sin(th), CART_Y + pendulum.l * np.cos(th))
    cartxy_visible = (x-CART_DISPLAY_WIDTH*.5, CART_Y)
    cartxy_true = (x, CART_Y)
    mass.set_center(massxy)
    cart.set_xy(cartxy_visible)
    line.set_xy((massxy, cartxy_true))
    time_text.set_text("t="+str(df.index.values[i]))
    return [mass, cart, time_text, line]

animation = FuncAnimation(fig, animate, df.count, init_func=init, blit=True, interval= .01)
plt.show()
'''
# Plot: x, theta, xdot, thetadot
ax2 = fig.add_subplot(gs[0,2:])
ax2.set_xlabel("t")
ax2.plot(solution.t, solution.y[0], label="x")
ax2.plot(solution.t, solution.y[1], label="theta")
ax2.plot(solution.t, solution.y[2], label="x dot")
ax2.plot(solution.t, solution.y[3], label="theta dot")
ax2.legend()

# Plot: forces, energy, momentum
ax3 = fig.add_subplot(gs[1,2:])
force_t = [f[0] for f in force_output]
force_forces = [f[1] for f in force_output]
ax3.scatter(force_t, force_forces, label='forces')
energy_t = [e[0] for e in energy]
energy_energies = [e[1] for e in energy]
# ax3.plot(energy_t, energy_energies, label='energy')
momentum_t = [m[0] for m in momentum]
momentum_cart = [m[1] for m in momentum]
momentum_pend = [m[2] for m in momentum]
momentum_total = [(m[1] + m[2]) for m in momentum]
# ax3.plot(momentum_t, momentum_cart, label='cart momentum')
# ax3.plot(momentum_t, momentum_pend, label='pendulum momentum')
# ax3.plot(momentum_t, momentum_total, label='total momentum')
ax3.legend()

animation = FuncAnimation(fig, animate, len(solution.y[0]), init_func=init, blit=True, interval=(1000/S_TPS))
# animation.save('video.mp4', fps=30)
plt.show()
'''