import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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
    def __init__(self, init_state, pendulum, T, dt, u_max=100, plotting=False):
        self.pendulum = pendulum 
        self.plotting = plotting
        self.setpoint = 0
        
        # System A 
        # nxn
        # 
        # d=damping
        # 
        # [ 0     1        0              0 ]
        # [ 0     -d/M     mg/M           0 ]
        # [ 0     0        0              1 ]
        # [ 0     -d/M*l   -(m+M)g/(M l)  0 ]  

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pendulum.g * pendulum.m)/pendulum.M, 0],
            [0, 0, 0, 1],
            [0, 0, pendulum.g/pendulum.l + pendulum.g * pendulum.m/(pendulum.l*pendulum.M), 0]
        ])

        # Input B
        # n x p
        # 
        # B = [ 0     ]
        #     [ 1/M   ]
        #     [ 0     ]
        #     [ 1/M*l ]

        B = np.array([
            [0], 
            [1/pendulum.M], 
            [0], 
            [1/(pendulum.M * pendulum.l)]
        ])

        # Output C
        # q x n
        # 
        # (blank for now)
        C = np.zeros((1, A.shape[0]))

        # Feedthrough D
        # q x p
        # 
        # (blank for now)
        D = np.zeros((1, 1))

        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A = sys_disc[0]
        self.B = sys_disc[1]
        
        self.T = T
        self.u_max = u_max

        self.planned_u = []
        self.planned_state = []

    def policy(self, state, t, dt):
        x = cp.Variable((4, self.T+1))
        u = cp.Variable((1, self.T))
        cost = 0
        constr = [x[:, 0] == state]
        for t in range(self.T):
            constr.append(x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t])
            constr.append(cp.abs(u[0, t]) <= self.u_max)
        
        cost += cp.sum_squares(cp.abs(x[2, :])) + cp.sum_squares(cp.abs(x[3,:]))

        problem = cp.Problem(cp.Minimize(cost), constraints=constr)
        problem.solve(verbose=False, solver='SCS')
        action = u[0,0].value

        # dump estimate info
        self.planned_state = x[:,:].value
        self.planned_u = u[0,:].value

        print('u: {}, theta: {}'.format(round(u[0,0].value,4),round(x[2,0].value,4)))
        return action       

    def init_plot(self):
        prediction = plt.figure()
        return prediction

    def update_plot(self, figure):
        plt.clf()
        ax0 = figure.add_subplot(211)
        ax1 = figure.add_subplot(212)
        # States
        ax0.plot(self.planned_state[0], label=r'$x$')
        ax0.plot(self.planned_state[1], label=r'$\dot{x}$') 
        ax0.plot(self.planned_state[2], label=r'$\theta$')
        ax0.plot(self.planned_state[3], label=r'$\dot{\theta}$')   
        ax0.axhline(y = self.setpoint, color='k', drawstyle='steps', linestyle='dotted', label=r'set point')
        ax0.set_ylabel("state")
        ax0.legend()
        # Control Actions
        ax1.plot(self.planned_u, label=r'$u$')
        ax1.set_ylabel("force")
        ax1.legend()
        plt.draw()
        plt.pause(0.001)


class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0

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

class PController(Controller):
    def __init__(self, x_0, setpoint, pgain, u_max):
        self.x_0 = x_0
        self.setpoint = setpoint
        self.pgain = pgain
        self.u_max = u_max
    def policy(self, x, t, dt):
        action = 0
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