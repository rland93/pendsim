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

class Visualizer(object):
    def __init__(self, data, pendulum, save=False, cart_display_width=2, cart_height=1, viz_limit=10, viz_size=10):
        '''
        Viz refers to the animation visualization
        '''
        self.data = data
        self.pendulum = pendulum
        self.save = save
        self.cart_display_width = cart_display_width # cart width
        self.cart_height = cart_height # cart height
        self.viz_limit = viz_limit # x, y limits
        self.viz_size = viz_size

    def display_viz(self):
        viz = plt.figure(figsize=(self.viz_size, self.viz_size))
        ax = plt.axes()
        ax.set_xlim(-self.viz_limit, self.viz_limit)
        ax.set_ylim(-self.viz_limit, self.viz_limit)
        cart = patches.Rectangle((-self.cart_display_width * 0.5, self.cart_height), width=self.cart_display_width, height=-self.cart_height, ec='black', fc='salmon')
        mass = patches.Circle((0,0), radius=np.sqrt(5)*0.14, fc='skyblue', ec='black')
        line = patches.FancyArrow(0,0,1,1)
        time_text = text.Annotation('', (4,4), xycoords='axes points')
        def init():
            ax.add_patch(cart)
            ax.add_patch(mass)
            ax.add_patch(line)
            ax.add_artist(time_text)
            return [mass, cart, time_text, line]

        # matplotlib animate doesn't play nice with dataframes :(
        animate_x = data['x'].values.tolist()
        animate_theta = data['theta'].values.tolist()
        animate_times = data.index.values.tolist()
        frames = len(data.index)
        def animate(i):
            x = animate_x[i] # position
            th = animate_theta[i] # angle
            massxy = (x + self.pendulum.l * np.sin(th), self.cart_height + self.pendulum.l * np.cos(th))
            cartxy_visible = (x - self.cart_display_width*.5, self.cart_height)
            cartxy_true = (x, self.cart_height)
            mass.set_center(massxy)
            cart.set_xy(cartxy_visible)
            line.set_xy((massxy, cartxy_true))
            time_text.set_text("t="+str(animate_times[i]))
            return [mass, cart, time_text, line]
        animation = FuncAnimation(viz, animate, frames, init_func=init, blit=True, interval= .1)
        if self.save:
            animation.save('video.mp4', fps=30)
        plt.show()
    
    def display_plots(self):
        figure = plt.figure()
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)
        ax1.plot(self.data['x'],label=r'$x$')
        ax1.plot(self.data['xdot'],label=r'$\dot{x}$')
        ax1.plot(self.data['theta'],label=r'$\theta$')
        ax1.plot(self.data['thetadot'],label=r'$\dot{\theta}$')
        ax1.legend()
        
        ax2.plot(self.data['cart momentum'], label='cart momentum')
        ax2.plot(self.data['pendulum momentum'], label='pendulum momentum')
        ax2.plot(self.data['total momentum'], label='total momentum')
        ax2.plot(self.data['energy'], label='energy')
        ax2.legend()

        return figure


if __name__ == "__main__":
    pendulum = Pendulum(10,5,5,9.81, np.array([0,0,0.5,0]))
    sim = Simulation(pendulum, .01, 5,[(1,0.5,2.5)])
    data = sim.simulate()
    print(data)

    plot = Visualizer(data, pendulum, save=False)
    plot.display_viz()
    a = plot.display_plots()
    plt.show()