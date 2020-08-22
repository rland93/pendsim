import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel as C
import matplotlib.animation as animation

class Controller(object):
    '''
    Class template for pendulum controller
    '''
    def __init__(self, init_state):
        self.init_state = init_state
    
    def policy(self, state):
        '''
        A controller must have a policy action.
        
        Parameters
        ----------
        state: (float, float, float float)
            The current system state
        
        Returns
        -------
        float
            The controller action, in force applied to the cart.
        '''
        raise NotImplementedError

class MPCController(Controller):
    def __init__(self, init_state, pendulum, T, u_max=15):
        self.pendulum = pendulum 
        
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pendulum.g * pendulum.m)/pendulum.M, 0],
            [0, 0, 0, 1],
            [0, 0, -(pendulum.m + pendulum.M) * pendulum.g / (pendulum.M * pendulum.l), 0]
        ])
        # The open-loop dynamic matrix A:
        # (linearized at theta = 0)
        ##### Linear Model Params #####
        # d = damping
        # [ 0     1        0              0 ]
        # [ 0     -d/M     mg/M           0 ]
        # [ 0     0        0              1 ]
        # [ 0     -d/M*l   -(m+M)g/(M l)  0 ]

        self.B = np.array([[0], [1/self.pendulum.M], [0], [1/(self.pendulum.M*self.pendulum.l)]])
        # The closed loop control matrix B:
        # (linearized at theta = 0)

        # B = [ 0     ]
        #     [ 1/M   ]
        #     [ 0     ]
        #     [ 1/M*l ]

        self.T = T
        self.u_max = 10000


    def policy(self, state, t, dt):
        x = cp.Variable((4, self.T+1))
        u = cp.Variable((1, self.T))
        cost = 0
        constr = []
        for t in range(self.T):
            constr += [x[:, t+1] == x[:,t] + dt * self.A @ x[:, t] + dt * self.B @ u[:, t],
            cp.norm_inf(u[:, t]) <= self.u_max]
        constr += [
            x[:, 0] == state,
            ]
        cost += cp.sum_squares(x[2, :])
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False)
        print('optimal u={}'.format(u[0,0].value))
        action = u[0,0].value
        # plt.plot(x[0,:].value)
        # plt.plot(x[2,:].value)
        # plt.plot(u.value)
        # plt.show()
        return action

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t):
        return 0
'''
I want to animate the controller, but it's a pain, so shelved.

class Viewer(object):
    def __init__(self):
        self.fig, self.lines = self.initialize()
    
    def initialize(self):
        plt.ion()
        fig = plt.figure()
        eps_true = plt.plot([],[], 'r.', markersize=10, label=r'$\epsilon$ (true)')
        eps_est = plt.plot([],[], 'b-', label=r'$\epsilon$ (predicted)')
        eps_range_upper = plt.plot([],[],fc='royalblue', linestyle=':', label='95% CI (upper)')
        eps_range_lower = plt.plot([],[],fc='royalblue', linestyle=':', label='95% CI (lower)')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\epsilon$')
        lines = [eps_est, eps_range_upper, eps_range_lower, eps_true]
        return fig, lines

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def render(self, eps_est_t, eps_est, eps_est_upper, eps_est_lower, eps_true_t, eps_true):
        self.lines[0].set_xdata(eps_est_t)
        self.lines[0].set_ydata(eps_est)

        self.lines[1].set_xdata(eps_est_t)
        self.lines[1].set_ydata(eps_est_upper)

        self.lines[2].set_xdata(eps_est_t)
        self.lines[2].set_ydata(eps_est_lower)

        self.lines[3].set_xdata(eps_true_t)
        self.lines[3].set_ydata(eps_true)

        adjust

        self.update()
'''


class MPCWithGPR(Controller):
    def __init__(self, window_size, pendulum):
        self.tick = 0
        self.window_size = window_size # number of readings to look backwards
        self.t_priors = [] # time priors
        self.x_priors = [] # state priors
        self.u_priors = [] # control priors
        self.m_eval = [] # linear model evaluated at priors
        self.e_priors = np.zeros((4, window_size)) # epsilon priors
        self.fig = plt.figure()

        ##### Linear Model Params #####
        # d = damping
        # [ 0     1        0              0 ]
        # [ 0     -d/M     mg/M           0 ]
        # [ 0     0        0              1 ]
        # [ 0     -d/M*l   -(m+M)g/(M l)  0 ]
        self.model_A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pendulum.g * pendulum.m)/pendulum.M, 0],
            [0, 0, 0, 1],
            [0, 0, -(pendulum.m + pendulum.M)/(pendulum.M * pendulum.l), 0]
        ])

        self.model_B = np.array([[0], [1/pendulum.M], [0], [1/(pendulum.M * pendulum.l)]])
        

    def policy(self, state, t, dt):
        # begin after window has passed so we have data
        # to look back on
        if self.tick > self.window_size:
            print("tick: {}".format(self.tick))
            # Keep the window at a manageable size
            self.t_priors.pop(0)
            self.x_priors.pop(0)
            self.u_priors.pop(0)
            e = np.zeros((4, self.window_size))

            # evaluate difference between linear model and 
            # real values for  each state in the window
            for i, (x, u) in enumerate(zip(self.x_priors, self.u_priors)):
                x = np.array(x, ndmin=2).transpose()
                if i == 0:
                    pass
                else:
                    # x_k+1 = Ax+Bu
                    # therefore when we evaluate epsilon priors, we take from
                    # the second item in the list of priors... not the first
                    # because we cannot find e = g(x_k,u_k) without knowing
                    # x_k-1, u_k-1
                    xk_true = self.x_priors[i]
                    dx = self.model_A @ self.x_priors[i - 1]
                    xk_model = self.x_priors[i - 1]  + dt * dx
                    # NOTE: There is never an error in either x or in theta,
                    # because the way we calculate x_k+1 is just x_k + xdot * dt;
                    # it's the same in our linear model as it is in the actual
                    # simulation. Since we measure xdot directly, we're never
                    # wrong! 
                    e[:, i] = np.array((xk_true - xk_model))

            e_td = e[1 ,1:]
            ts = np.array(self.t_priors[1:], ndmin=2)
            print("shape(e) = {}, shape(t) = {}".format(np.shape(e_td), np.shape(ts)))
            kernel = C(1) * RBF(length_scale=(dt * self.window_size/2))

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
            gp.fit(ts.transpose(), e_td)
            
            t_pred = np.atleast_2d(np.linspace(self.t_priors[0], self.t_priors[-1], 100)).transpose()

            pred, sigma = gp.predict(t_pred, return_std=True)

            # set plot elements
            plt.close()
            plt.title("$\dot{x}$")
            plt.plot(ts.transpose(), e_td, 'r.', markersize=10, label=r'$\epsilon$')
            plt.plot(t_pred, pred, 'b-', label=r'$\epsilon$ (predicted)')
            plt.fill(np.concatenate([t_pred, t_pred[::-1]]),
                    np.concatenate([pred - 1.9600 * sigma,
                                    (pred + 1.9600 * sigma)[::-1]]),
                    alpha=.3, fc='royalblue', ec='None', label='95% confidence interval')
            plt.xlabel(r'$t$')
            plt.ylabel(r'$\epsilon$')
            plt.ylim(min(pred - 1.96 * sigma) - .01, max(pred + 1.96 * sigma) + 0.01)
            plt.legend()
            plt.savefig('./plot_anim/' + str(self.tick) + '.png')

            action = 0
        else:
            action = 0
        
        self.t_priors.append(t)
        self.x_priors.append(list(state))
        self.u_priors.append(action)
        self.tick += 1
        return action
        
class MPCOneShot(Controller):
    def __init__(self):
        self.ts = []
        self.states = []
        self.tick = 0
    
    def policy(self, state, t):
        self.tick += 1
        self.states.append(list(state))
        self.ts.append(t)
        return 0

class PDController(Controller):
    def __init__(self, init_state, setpoint, pgain, dgain, igain, control_max):
        '''
        Paramters
        ---------
        init_state: 
            the initial state
        setpoint:
            desired theta val.
        pgain:
            proportional gain
        dgain:
            derivative gain
        igain:
            integral gain
        control_max:
            maximum controller action
        '''
        self.init_state = init_state
        self.setpoint = setpoint
        self.pgain = pgain
        self.dgain = dgain
        self.igain = igain
        self.control_max = control_max
        self.prior_state = 0 
        self.prior_t = 0
        self.prior_th = 0
        self.integral = 0
    def sign(self, x):
        if x > 0:
            return 1
        else:
            return -1

    def policy(self, state, t):
        '''
        Parameters
        ----------
        state: (float, float, float, float)
            Current system state
        t: float
            Current system time
        '''
        error = state[2] - self.setpoint

        dt = t - self.prior_t
        if dt > 0:
            deriv = (state[2] - self.prior_th)/(t - self.prior_t)
        else: 
            deriv = 0

        action = error * self.pgain + deriv * self.dgain

        self.prior_t = t
        self.prior_th = state[2] 
        return action

class BangBang(Controller):
    def __init__(self, setpoint, magnitude):
        self.setpoint = setpoint
        self.magnitude = magnitude
    
    def policy(self, state, t):
        error = state[2] - self.setpoint
        if error > 0.1 and state[2] < np.pi/4:
            return self.magnitude
        elif error < -0.1 and state[2] > -np.pi/4:
            return -self.magnitude
        else:
            return 0