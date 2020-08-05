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
    def __init__(self, M, m, l, g, init_state=(0,0,0,0)):
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
        self.init_state = init_state
    
    def calculate_accel(self, state, u):
        '''
        Update accelerations xddot, thetaddot from current state, given an input force.

        Parameters
        ----------
        u: float
            Force on the cart pointing right, in newtons.
        
        Returns
        -------
        (float, float)
            xdd and thetadd (the acceleration)
        '''
        # state =   [x, xdot, theta, thetadot]
        sin_t = np.sin(state[2])
        cos_t = np.cos(state[2])

        # A^-1     2x2
        A = np.linalg.inv(
            np.array([
            [self.M + self.m,       -self.m * self.l * cos_t],
            [-cos_t,                self.l]
            ])
        )

        # B  2 @ 1
        B = np.array([
            [-self.m*self.l*state[3]*state[3] * sin_t + u],
            [self.g * sin_t]
        ])

        solution = A @ B
        xdd = solution[0][0]
        tdd = solution[1][0]
        return (xdd, tdd)

    def pend_update(self, state, dt, u):
        '''
        Given the current state, timestep, and force applied, updates the pendulum state over that timestep.

        Parameters
        ----------
        state: (float, float, float, float)
            The current state
        dt: float
            The length of the update step (seconds, I guess?)
        u: float
            The force
        
        Returns
        -------
        (float, float, float, float)
            The new state after time interval dt [x, xdot, theta, thetadot]
        '''
        # update acceleration xddot_k, thddot_k = f(x_k, u_k)
        (xdd, tdd) = self.calculate_accel(state, u)
        # write accelerations
        self.xdd = xdd
        self.tdd = tdd

        # Euler method, x_k+1 = x_k + xdot_k * dt
        x = state[0] + state[1] * dt
        xd = state[1] + xdd * dt
        t = state[2] + state[3] * dt
        td = state[3] + tdd * dt

        return (x, xd, t, td)


    def calculate_state_energy(self, state):
        '''
        Calculate the energy of the current state

        Parameters
        ----------
        state: (float, float, float, float)
            The current state

        Returns
        -------
        (float float):
            The kinetic and potential energy of the current state
        '''
        # cart vel. squared
        v_c2 = state[1] * state[1] 
        # pendulum velocity squared
        v_p2 = v_c2 - 2 * self.l * state[1] * state[3] * np.cos(state[2]) + self.l * self.l * state[3] * state[3]

        KE = 0.5 * self.M * v_c2 + .5 * self.m * v_p2
        PE = self.m * self.g * self.l * np.cos(state[2])

        return (KE, PE)

    def calculate_momentum(self, state):
        '''
        Calculate the (x-direction) linear momentum of the current state.

        Paramters
        ---------
        state: (float, float, float, float)
            The current state

        Returns
        -------
        (float, float):
            The current momentums of the cart and the pendulum.
        '''
        p_cart = self.M * state[1] 
        p_pend = self.m * (state[1] + state[3] * np.sin(state[2]))
        return (p_cart, p_pend)

class Simulation(object):
    '''
    The simulation object.
    '''
    def __init__(self, pendulum, dt, t_final, u):
        '''
        Parameters
        ----------
        pendulum: `Pendulum`
            The pendulum to be simulated
        dt: float
            The time step of the simulation
        t_final: float
            Simulate up to this time
        u: list of (float, float, float)
            List of force magnitude, start time, and end time pairs.
        k: int
            The kth simulation timestep, starting from 0.
        times: [int]
            List of simulation times of length k.
        data: dict
            Dictionary containing the simulation data. Keys are the attribute,
            values are the data at each step of the simulation.
        '''
        self.pendulum = pendulum
        self.dt = dt
        self.t_final = t_final
        self.u = u
        self.k = 0
        self.times = []
        self.data={
            'x': [], 
            'xdot': [],
            'xdd' : [],
            'theta':[],
            'thetadot':[],
            'tdd': [],
            'PE':[], 
            'KE':[], 
            'E':[], 
            'cart momentum': [],
            'pendulum momentum': [], 
            'total momentum': [],
            'forces' : []}
    
    def simulate(self):
        '''
        Run the simulation.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing simulation data, indexed by time.
        '''
        t = 0
        state = pendulum.init_state
        while t <= self.t_final:
            # record the current state, energies, etc.
            self.times.append(t)
            self.data['x'].append(state[0])
            self.data['xdot'].append(state[1])
            self.data['xdd'].append(self.pendulum.xdd)
            self.data['theta'].append(state[2])
            self.data['thetadot'].append(state[3])
            self.data['tdd'].append(self.pendulum.tdd)
            self.data['KE'].append(self.pendulum.calculate_state_energy(state)[0])
            self.data['PE'].append(self.pendulum.calculate_state_energy(state)[1])
            self.data['E'].append(self.pendulum.calculate_state_energy(state)[0] + self.pendulum.calculate_state_energy(state)[1])
            p_cart, p_pend = self.pendulum.calculate_momentum(state)
            self.data['cart momentum'].append(p_cart) 
            self.data['pendulum momentum'].append(p_pend)
            self.data['total momentum'].append(p_cart + p_pend)

            # add up all present forces
            u_k = 0
            for force in self.u:
                f_begin = force[1]
                f_end = force[1] + force[2]
                if f_begin < t < f_end:
                    u_k+= force[0]
            self.data['forces'].append(u_k)
            
            # update state
            state = pendulum.pend_update(state, self.dt, u_k)
            t += self.dt

        return pd.DataFrame(index=self.times, data=self.data)

class Visualizer(object):
    '''
    Visualizer object is what we use to visualize the animated pendulum as well as display plots.
    '''
    def __init__(self, data, pendulum, frameskip=1, save=False, cart_display_width=2, cart_height=1, viz_size=(10,10), viz_window_size=(16, 9)):
        '''
        Parameters
        ----------
        data : pandas.DataFrame
            The data from the simulation
        pendulum : `Pendulum`
            The pendulum that is visualized
        frameskip : int, default 1
            The number of calculated time intervals to skip. Setting frameskip 
            to 1 will show every frame, 3 will show only every 3rd frame, etc. 
            This allows to run the animation at faster than real time speed.
        save : bool, default False
            Whether to save the animation. Saving requires ffmpeg to function.
        cart_display_width : float, default 1
            The display width of the cart as it appears on the animation.
        cart_height : float, default 1
            The display height of the cart as it appears on the animation.
        viz_size : (float, float)
            The size of the visualized canvas
        viz_window_size : (float, float)
            The size of the visualization window (or of the video once rendered)
        '''
        self.data = data
        self.pendulum = pendulum
        self.frameskip = frameskip
        self.save = save
        self.cart_display_width = cart_display_width
        self.cart_height = cart_height 
        self.viz_size = viz_size 
        self.viz_window_size = viz_window_size

    def display_viz(self):
        '''
        Display (show) the animated visualization. This function calls plt.show()

        Returns
        -------
        None
        '''
        viz = plt.figure(figsize=self.viz_window_size)
        ax = plt.axes()
        plt.axis('scaled')
        ax.set_xlim(-self.viz_size[0], self.viz_size[0])
        ax.set_ylim(-self.viz_size[1], self.viz_size[1])
        cart = patches.Rectangle((-self.cart_display_width * 0.5, self.cart_height), width=self.cart_display_width, height=-self.cart_height, ec='black', fc='seagreen')
        mass = patches.Circle((0,0), radius=np.sqrt(5)*0.14, fc='skyblue', ec='black')
        line = patches.FancyArrow(0,0,1,1)
        force = patches.FancyArrow(0,0,1,1, ec='red')
        time_text = text.Annotation('', (4,4), xycoords='axes points')
        ground = patches.Rectangle((-1000, -2000), 2000, 2000, fc='lightgrey')
        ax.add_patch(ground)

        def init():
            ax.add_patch(cart)
            ax.add_patch(mass)
            ax.add_patch(line)
            ax.add_artist(time_text)
            ax.add_patch(force)
            return [mass, cart, time_text, line, force]

        # matplotlib animate doesn't play nice with dataframes :(
        animate_x = data['x'].values.tolist()[::self.frameskip]
        animate_theta = data['theta'].values.tolist()[::self.frameskip]
        animate_force = data['forces'].values.tolist()[::self.frameskip]
        animate_times = data.index.values.tolist()[::self.frameskip]
        frames = len(animate_times)

        def animate(i):
            x = -animate_x[i] # position
            th = animate_theta[i] # angle
            u = animate_force[i] # force
            massxy = (x + self.pendulum.l * np.sin(th), self.cart_height + self.pendulum.l * np.cos(th))
            cartxy_visible = (x - self.cart_display_width*.5, self.cart_height)
            # animate force application
            if u > 0.0:
                force_begin = (x + .5 * self.cart_display_width, .5 * self.cart_height)
                force_end = (x + .5 * self.cart_display_width + np.sqrt(.1*u), .5 * self.cart_height)
                force.set_xy((force_begin, force_end))
                force.set_linewidth(np.sqrt(u))
                force.set_visible(True)
            elif u < 0.0:
                force_begin = (x - .5 * self.cart_display_width, .5 * self.cart_height)
                force_end = (x - .5 * self.cart_display_width - np.sqrt(.1*np.abs(u)), .5 * self.cart_height)
                force.set_xy((force_begin, force_end))
                force.set_linewidth(np.sqrt(np.abs(u)))
                force.set_visible(True)
            else: 
                force.set_visible(False)

            cartxy_true = (x, self.cart_height)
            mass.set_center(massxy)
            cart.set_xy(cartxy_visible)
            line.set_xy((massxy, cartxy_true))
            time_text.set_text("t="+str(animate_times[i]))
            return [mass, cart, time_text, line, force]
        animation = FuncAnimation(viz, animate, frames, init_func=init, blit=True, interval=32)
        if self.save:
            animation.save('video.mp4', fps=30)
        plt.show()
    
    def display_plots(self):
        '''
        Return a figure containing the plots, which can then be saved or displayed.

        Returns
        -------
        `matplotlib.pyplot.figure`
        '''
        figure = plt.figure()
        ax0 = figure.add_subplot(411)
        ax1 = figure.add_subplot(412)
        ax2 = figure.add_subplot(413)
        ax3 = figure.add_subplot(414)
        ax0.plot(self.data['x'],label=r'$x$')
        ax1.plot(self.data['xdot'],label=r'$\dot{x}$')
        ax0.plot(self.data['theta'],label=r'$\theta$')
        ax1.plot(self.data['thetadot'],label=r'$\dot{\theta}$')
        ax1.plot(self.data['xdd'],label=r'xdd')
        ax1.plot(self.data['tdd'],label=r'tdd')
        ax1.legend()
        # ax2.plot(self.data['cart momentum'], label='cart momentum')
        # ax2.plot(self.data['pendulum momentum'], label='pendulum momentum')
        # ax2.plot(self.data['total momentum'], label='total momentum')
        ax2.plot(self.data['PE'], label='PE')
        ax2.plot(self.data['KE'], label='KE')
        ax2.plot(self.data['E'], label='E')
        ax2.legend()
        ax3.plot(self.data['forces'], label='force')

        return figure


if __name__ == "__main__":
    forces = [
        (50, 1, 0.5),
        (-100, 10, 0.5),
        (-50, 16, .5),
        (200, 19, .25)
    ]
    pendulum = Pendulum(15,2,3,9.81, np.array([0,0,0.1,0]))
    sim = Simulation(pendulum, .001, 30, forces)
    data = sim.simulate()
    plot = Visualizer(data, pendulum, frameskip=32, save=False, viz_size=(20,8), viz_window_size=(16,9))
    plot.display_viz()
    plot.display_plots()
    plt.show()