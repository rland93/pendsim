import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.style
matplotlib.style.use('seaborn-deep')
import controller

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

############## PENDULUM #############

class Pendulum(object):
    '''
    An inverted pendulum object.
    '''
    def __init__(self, M, m, l, g, x_0=np.array([0,0,0,0]), cfric=0.0075, pfric=0.0075):
        '''
        M: cart mass
        m: ball mass
        l: pend length
        g: gravity
        cfric: cart (viscous) friction
        pfric: pend (viscout) friction
        x_0: initial state
        '''
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.cfric = cfric
        self.pfric = pfric
        self.x_0 = x_0

    def calculate_xdot(self, x, u):
        '''
        Calculate xdot = f(x, u) where x is the state vector 
        xdot is a vector [xdot, xddot, tdot, tddot] It's may
        be a little bit confusing because xdot indicates either
        the state vector or the cart absolute position/velo
        depending on context.
        '''
        # state =   [x, xdot, theta, thetadot]
        sin_t = np.sin(x[2])
        cos_t = np.cos(x[2])
        # A^-1     2x2
        A = np.linalg.inv(
            np.array([
            [self.M + self.m,       -self.m * self.l * cos_t],
            [-cos_t,                self.l]
            ], dtype='float')
        )
        # B  2 @ 1
        B = np.array([
            [-self.m*self.l*x[3]*x[3] * sin_t + u],
            [self.g * sin_t]
        ])
        solution = A @ B

        xd = x[1]
        xdd = solution[0][0] - x[1] * self.cfric
        td = x[3]
        tdd = solution[1][0] - x[3] * self.pfric

        return np.array([xd, xdd, td, tdd])

    def update_rk4(self, x, u, dt):
        '''
        Update the pendulum state using the rk4 method
        '''
        k1 = self.calculate_xdot(x, u) * dt
        k2_state = x + k1 * 0.5 * dt
        k2 = self.calculate_xdot(k2_state, u)
        k3_state = x + k2 * 0.5 * dt
        k3 = self.calculate_xdot(k3_state, u)
        k4_state = x + k3 * dt
        k4 = self.calculate_xdot(k4_state, u)
        state = x + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt
        # wrap pi
        # state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        return state

    def get_energy(self, x):
        '''
        Get a tuple containing the kinetic and potential energy of the system.
        '''
        # cart vel^2
        v_c2 = x[1] * x[1]
        # pend vel^2
        v_p2 = v_c2 - 2 * self.l * x[1] * x[3] * np.cos(x[2]) + self.l * self.l * x[3] * x[3]
        ke = 0.5 * self.M * v_c2 + 0.5 * self.m * v_p2
        pe = self.m * self.g * self.l * np.cos(x[2])
        return ke, pe, ke + pe

    def get_momentum(self, x):
        '''
        Calculate the (x-direction) linear momentum of the system.
        '''
        p_cart = self.M * x[1] 
        p_pend = self.m * (x[1] + x[3] * np.sin(x[2]))
        return p_cart, p_pend

class Simulation(object):
    '''
    The simulation object. Provide a pendulum, a timestep, a final time, and a list of 
    external forces.
    '''
    def __init__(self, pend, dt, t_final, f_func, control_every, noise_scale):
        self.pend = pend # pendulum to be simulated
        self.dt = dt # time step
        self.t_final = t_final # end at or before this time
        self.f_func = f_func # forcing function
        self.times = [] # list of discrete points in time simulated
        self.control_every = control_every # control action interval, 1 = control every dt, 2 = control every other dt, etc.
        self.noise_scale = noise_scale # noise given to the state. can be scalar (equal noise) or len 4 array (noise given to each state)
        self.data={
            'x': [], 
            'xdot': [],
            'theta':[],
            'thetadot':[],
            'PE':[], 
            'KE':[], 
            'E':[], 
            'cart momentum': [],
            'pendulum momentum': [], 
            'total momentum': [],
            'forces' : [],
            'control action' : [],
            'mu_x': [],
            'mu_xd': [],
            'mu_t': [],
            'mu_td': [],
            'sigma_x': [],
            'sigma_xd': [],
            'sigma_t': [],
            'sigma_td': [],
            'nl_pred_x' : [],
            'nl_pred_xd' : [],
            'nl_pred_t' : [],
            'nl_pred_td' : [],
            'nl_error_x' : [],
            'nl_error_xd' : [],
            'nl_error_t' : [],
            'nl_error_td' : [],
            'conf_lower_x' : [],
            'conf_lower_xd' : [],
            'conf_lower_t' : [],
            'conf_lower_td' : [],
            'conf_upper_x' : [],
            'conf_upper_xd' : [],
            'conf_upper_t' : [],
            'conf_upper_td' : [],
            'l_pred_x' : [],
            'l_pred_xd' : [],
            'l_pred_t' : [],
            'l_pred_td' : [],
            'l_error_x' : [],
            'l_error_xd' : [],
            'l_error_t' : [],
            'l_error_td' : [],
            } # data returned by the sim
    
    def simulate(self, controller):
        '''
        Run the simulation with the given controller
        '''
        # initialize
        t_k = 0
        x_k = self.pend.x_0
        # no of runs
        n = 0

        # controller plots
        if controller.plotting:
            figure = controller.init_plot()

        # step time
        while t_k <= self.t_final:
            # print('time={}, x_k={}'.format(round(t_k,3), x_k))
            # forces
            u_k = self.f_func(t_k)

            # controller takes action every `control_every` steps
            if n % self.control_every == 0:

                noise_on_state = x_k + np.random.random_sample(np.shape(x_k)) * self.noise_scale

                action = controller.policy(noise_on_state, t_k, self.dt)
                if controller.plotting:
                    controller.update_plot(figure)
            
            # write data
            self.write_data_timestep(x_k, t_k, u_k, action, controller.l_pred_x_k, controller.l_err_x_k, controller.nl_pred_x_k, controller.nl_err_x_k, controller.pred_sig, controller.pred_mu)

            # add action to extern. force to get total force
            u_k += action

            # update state, time
            x_k = self.pend.update_rk4(x_k, u_k, self.dt)
            t_k += self.dt
            n += 1
        return pd.DataFrame(index = self.times, data = self.data)
    
    def write_data_timestep(self, x_k, t_k, u_k, action, l_pred_x_k, l_err_x_k, nl_pred_x_k, nl_err_x_k, sigma, mu):
        '''
        write recorded data for a single timestep
        '''
        # times
        self.times.append(t_k)
        # states
        self.data['x'].append(x_k[0])
        self.data['xdot'].append(x_k[1])
        self.data['theta'].append(x_k[2])
        self.data['thetadot'].append(x_k[3])
        # energies
        ke, pe, e = self.pend.get_energy(x_k)
        self.data['KE'].append(ke)
        self.data['PE'].append(pe)
        self.data['E'].append(e)
        # momentums
        p_cart, p_pend = self.pend.get_momentum(x_k)
        self.data['cart momentum'].append(p_cart) 
        self.data['pendulum momentum'].append(p_pend)
        self.data['total momentum'].append(p_cart + p_pend)
        # forces
        self.data['forces'].append(u_k)
        # controller action
        self.data['control action'].append(action)

        # non-linear predictions & errors
        self.data['nl_pred_x'].append(nl_pred_x_k[0,0])
        self.data['nl_pred_xd'].append(nl_pred_x_k[0,1])
        self.data['nl_pred_t'].append(nl_pred_x_k[0,2])
        self.data['nl_pred_td'].append(nl_pred_x_k[0,3])
        self.data['nl_error_x'].append(nl_err_x_k[0,0])
        self.data['nl_error_xd'].append(nl_err_x_k[0,1])
        self.data['nl_error_t'].append(nl_err_x_k[0,2])
        self.data['nl_error_td'].append(nl_err_x_k[0,3])
        # mu
        self.data['mu_x'].append(mu[0,0])
        self.data['mu_xd'].append(mu[0,1])
        self.data['mu_t'].append(mu[0,2])
        self.data['mu_td'].append(mu[0,3])
        self.data['sigma_x'].append(sigma[0,0])
        self.data['sigma_xd'].append(sigma[0,1])
        self.data['sigma_t'].append(sigma[0,2])
        self.data['sigma_td'].append(sigma[0,3])



        self.data['conf_lower_x'].append(mu[0,0] - sigma[0, 0])
        self.data['conf_lower_xd'].append(mu[0,1] - sigma[0, 1])
        self.data['conf_lower_t'].append(mu[0,2] - sigma[0, 2])
        self.data['conf_lower_td'].append(mu[0,3] - sigma[0, 3])
        self.data['conf_upper_x'].append(mu[0,0] + sigma[0, 0])
        self.data['conf_upper_xd'].append(mu[0,1] + sigma[0, 1])
        self.data['conf_upper_t'].append(mu[0,2] + sigma[0, 2])
        self.data['conf_upper_td'].append(mu[0,3] + sigma[0, 3])

        # linear predictions & errors
        self.data['l_pred_x'].append(l_pred_x_k[0,0])
        self.data['l_pred_xd'].append(l_pred_x_k[0,1])
        self.data['l_pred_t'].append(l_pred_x_k[0,2])
        self.data['l_pred_td'].append(l_pred_x_k[0,3])
        self.data['l_error_x'].append(l_err_x_k[0,0])
        self.data['l_error_xd'].append(l_err_x_k[0,1])
        self.data['l_error_t'].append(l_err_x_k[0,2])
        self.data['l_error_td'].append(l_err_x_k[0,3])


    def wrap2pi(self, theta):
        '''
        wrap theta to the interval [0, 2pi]
        '''
        return np.arctan2(np.sin(theta), np.cos(theta))

################ VISUALIZATION ####################

class Visualizer(object):
    '''
    Visualizer object is what we use to visualize the animated pendulum as well as display plots.
    '''

    def __init__(self, data, pend, frameskip=10, save=False, filename='./video.mp4', cart_squish=2, window=(16,9)):
        # Sim Info
        self.data = data
        self.pend = pend
        
        # Movie Params
        self.save = save
        self.filename = filename
        
        # Playback Params
        self.skip = frameskip
        self.window = window

        ### DISPLAY ###
        self.disp_size = .5
        # Cart Params
        self.cart_squish = cart_squish
        self.cart_w = np.sqrt(self.cart_squish * self.pend.M) * self.disp_size
        self.cart_h = np.sqrt(1/(self.cart_squish) * self.pend.M) * self.disp_size
        # Pendulum Params
        self.p_rad = np.sqrt(self.pend.m) * self.disp_size/3
        # Display Params
        self.xmax = self.data['x'].max() * 1.1
        self.xmin = self.data['x'].min() * 1.1
        self.ymax = (self.pend.l + self.cart_h) * 1.3
        self.ymin = -self.pend.l * 1.3

    def initialize_objects(self):
        '''
        The initial draw of each of the objects
        '''
        # The "zero point" of the cart is physically where the pendulum connects.
        # So we adjust the animation position of the cart by moving it left by half
        # cart width and up by cart height
        cart = patches.Rectangle(
            (-self.cart_w * 0.5, self.cart_h), 
            width = self.cart_w, 
            height = -self.cart_h, 
            fc = 'seagreen',
            ec = 'black')
        # The pendulum mass
        mass = patches.Circle(
            (0,0), 
            radius=self.p_rad, 
            fc='skyblue', 
            ec='black')
        # The line connecting cart to pend mass
        line = patches.FancyArrow(0,0,1,1)
        # Line for external force
        ext_force = patches.FancyArrow(0,0,1,1, ec='red')
        # Line for control force
        ctrl_force = patches.FancyArrow(0,0,1,1, ec='blue')

        ground = patches.Rectangle((-1000, -2000), 2000, 2000, fc='lightgrey')
        ground.set_zorder(-1)

        # text
        angle_text = text.Annotation('', (4,4), xycoords='axes points')
        x_text = text.Annotation('',(4,16), xycoords='axes points')
        time_text = text.Annotation('', (4,28), xycoords='axes points')
        return cart, mass, line, ext_force, ctrl_force, ground, angle_text, x_text, time_text

    def display_viz(self):
        '''
        Display (show) the animated visualization. This function calls plt.show()
        '''
        # axis setup
        viz = plt.figure(figsize=self.window)
        ax = plt.axes()
        plt.axis('scaled')
        ax.set_xlim(self.xmin - self.pend.l*2, self.xmax + self.pend.l*2)
        ax.set_ylim(self.ymin, self.ymax)
    
        # matplotlib animate doesn't play nice with dataframes :(
        anim_x = data['x'].values.tolist()[::self.skip]
        anim_th = data['theta'].values.tolist()[::self.skip]
        anim_f = data['forces'].values.tolist()[::self.skip]
        anim_c = data['control action'].values.tolist()[::self.skip]
        anim_t = data.index.values.tolist()[::self.skip]
        n_frames = len(anim_t)
        # Initialize objects
        cart, mass, line, ext_force, ctrl_force, ground, angle_text, x_text, time_text = self.initialize_objects()

        def init():
            '''
            Function required by matplotlib. Initializes the objects for use by the animator
            '''
            ax.add_patch(cart)
            ax.add_patch(mass)
            ax.add_patch(line)
            ax.add_patch(ext_force)
            ax.add_patch(ctrl_force)
            ax.add_patch(ground)
            ax.add_artist(angle_text)
            ax.add_artist(x_text)
            ax.add_artist(time_text)
            return [ground, cart, mass, line, ext_force, ctrl_force, angle_text, x_text, time_text]

        def animate(i):
            '''
            Function required by matplotlib. Runs in a loop during FuncAnimation
            '''
            # draw extern force
            self.draw_force(ext_force, anim_f[i], anim_x[i], 0.6)
            # draw control force
            self.draw_force(ctrl_force, anim_c[i], anim_x[i], 0.3)
            # draw cart
            cartxy_true = (anim_x[i], self.cart_h)
            cartxy_visible = (anim_x[i] - self.cart_w * .5, self.cart_h)
            cart.set_xy(cartxy_visible)
            # draw pend mass
            # theta is formed by this triangle:
            # 
            #░   -lsin(theta) |
            #                 V
            # pend <x,y>░█▀▀▀▀▀█▀█░
            #           ░░█░░░░█▄█░
            #           ░░░█░░░░░█░
            #           ░░░░█░░░░█░
            #           ░░░░░█░th█ <- lcos(theta)░
            #           ░░░░░░█░░█░
            #           ░░░░░░░█░█░
            #           ░░░░░░░░██░
            #           ░░░░░░░░░█░cart <x,y>
            massxy = (anim_x[i] - self.pend.l * np.sin(anim_th[i]), self.cart_h + self.pend.l * np.cos(anim_th[i]))
            mass.set_center(massxy)
            # draw connecting line
            line.set_xy((massxy, cartxy_true))
            # display text
            angle_text.set_text(r"$\theta=$"+str(round(anim_th[i],3)))
            x_text.set_text(r"$x=$" + str(round(anim_x[i],3)))
            time_text.set_text(r"t="+str(round(anim_t[i],3)))
            return [ground, cart, mass, line, ext_force, ctrl_force, angle_text, x_text, time_text]
        
        def run_animation():
            '''
            Function to actually run the animation. Allows pausing on screen
            '''
            anim_running = True
            animation = FuncAnimation(viz, animate, frames=n_frames, init_func=init, blit=True, interval=16)
            def onClick(event):
                nonlocal anim_running
                if anim_running:
                    animation.event_source.stop()
                    anim_running = False
                else:
                    animation.event_source.start()
                    anim_running = True
            viz.canvas.mpl_connect('button_press_event', onClick)
            if self.save:
                animation.save('./video.mp4', fps=30, bitrate=1000)
        run_animation()
        plt.show()
    
    def draw_force(self, obj, u, cart_x, ydist):
        if u > 0.0:
            beg = (cart_x - .5 * self.cart_w, ydist * self.cart_h)
            end = (cart_x - .5 * self.cart_w - np.sqrt(.1 * np.abs(u)), ydist * self.cart_h)
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        elif u < 0.0:
            beg = (cart_x + .5 * self.cart_w, ydist * self.cart_h)
            end = (cart_x + .5 * self.cart_w + np.sqrt(.1 * np.abs(u)), ydist * self.cart_h)
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        else:
            obj.set_xy(((0,0), (1,1)))
            obj.set_linewidth(0)
            obj.set_visible(False)


    def display_plots(self):
        '''
        fig = plt.figure(figsize=(7.5, 10), tight_layout=True)
        gs = GridSpec(6, 1, figure=fig)
        labels = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
        lss = ['-', ':', '--', '-.']
        lcs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        # errors
        ax_states = fig.add_subplot(gs[0,0])
        ax_states.set_title('States')
        ax_states.plot(self.data['x'], ls=lss[0], c=lcs[0], label=labels[0])
        ax_states.plot(self.data['theta'], ls=lss[0], c=lcs[1], label=labels[1])
        ax_states.plot(self.data['xdot'], ls=lss[0], c=lcs[2], label=labels[2])    
        ax_states.plot(self.data['thetadot'], ls=lss[0], c=lcs[3], label=labels[3])
        ax_states.set_ylabel('Magnitude (m, rad)')
        ax_states.legend(loc=2)
        ax_forces = fig.add_subplot(gs[1,0], sharex=ax_states)
        ax_forces.set_title('Forces & Control Actions')
        ax_forces.plot(self.data['control action'],  ls=lss[0], c=lcs[0],  label='control action')
        ax_forces.plot(self.data['forces'],  ls=lss[1], c=lcs[0], label='external force')
        ax_forces.set_ylabel('Force applied (N)')
        ax_forces.legend(loc=2)
        ax_errorx = fig.add_subplot(gs[2,0], sharex=ax_states)
        ax_errorx.set_title("Error")
        ax_errorx.plot(self.data['l_error_x'], ls=lss[0], c=lcs[0], label=('l err: ' + labels[0]))
        ax_errorx.plot(self.data['nl_error_x'], ls=lss[1], c=lcs[0], label=('nl err: ' + labels[0]))
        ax_errorx.fill_between(self.data.index, self.data['conf_upper_x'], self.data['conf_lower_x'])
        ax_errorx.legend(loc=2)
        ax_errorxd = fig.add_subplot(gs[3,0], sharex=ax_states)
        ax_errorxd.plot(self.data['l_error_xd'], ls=lss[0], c=lcs[1], label=('l err: ' + labels[1]))
        ax_errorxd.plot(self.data['nl_error_xd'], ls=lss[1], c=lcs[1], label=('nl err: ' + labels[1]))
        ax_errorxd.legend(loc=2)
        ax_errort = fig.add_subplot(gs[4,0], sharex=ax_states)
        ax_errort.plot(self.data['l_error_t'], ls=lss[0], c=lcs[2], label=('l err: ' + labels[2]))
        ax_errort.plot(self.data['nl_error_t'], ls=lss[1], c=lcs[2], label=('nl err: ' + labels[2]))
        ax_errort.legend(loc=2)
        ax_errortd = fig.add_subplot(gs[5,0], sharex=ax_states)
        ax_errortd.plot(self.data['l_error_td'],  ls=lss[0], c=lcs[3], label=('l err: ' + labels[3]))
        ax_errortd.plot(self.data['nl_error_td'],  ls=lss[1], c=lcs[3], label=('nl_err: ' + labels[3]))
        ax_errortd.legend(loc=2)
        '''
        fig = plt.figure()
        state = fig.add_subplot(411)
        state.plot(self.data['theta'])
        forces = fig.add_subplot(412, sharex=state)
        forces.plot(self.data['forces'])
        forces.plot(self.data['control action'])
        pred = fig.add_subplot(413, sharex=state)
        pred.fill_between(self.data.index, self.data['conf_upper_t'], self.data['conf_lower_t'], facecolor='lightgray', alpha=0.5)
        pred.plot(self.data['conf_upper_t'], ':')
        pred.plot(self.data['conf_lower_t'], ':')
        pred.plot(self.data['mu_t'])
        errors = fig.add_subplot(414, sharex=state)
        errors.plot(self.data['l_error_t'])
        errors.plot(self.data['nl_error_t'])
        errors.fill_between(self.data.index, self.data['sigma_t'], facecolor='lightgray')
        plt.show()


if __name__ == "__main__":
   
    # period
    dt = 0.001
    time = 12
    # frequency
    dt_inv = 1/dt

    init = np.array([0,0,0,0])
    # M, m, l, g
    pnd = Pendulum(3, 1, 2, 9.81, x_0=init)
    
    # control action every n timesteps
    every = 10

    # ctrl action is .01
    # 100 -> 1 second

    # forces: (magnitude, start time, duration)
    # f_eqn = lambda x: 30 * np.cos(x)
    # f_ind = np.linspace(0, time, int(time*dt_inv))

    # starts = [round(m*dt, 3) for m in list(range(f_ind.shape[0]))]
    # mags = [f_eqn(x) for x in starts]
    # durations = [dt for m in list(range(f_ind.shape[0]))]

    # forces = list(zip(mags, starts, durations))

    forces = lambda x: 30 * np.cos(4*x) * np.exp(-0.5 * (x-4)**2)

    # measurement noise
    noise=0

    sim = Simulation(
        pnd, 
        dt, 
        time, 
        forces,
        every,
        noise)
    ctrl2 = controller.MPCWithGPR(
        pnd, 
        dt,
        window=7,
        every=5)

    data = sim.simulate(
        ctrl2)

    plot = Visualizer(
        data,
        pnd,
        frameskip = 30,
    )

    plot.display_viz()
    plot.display_plots()