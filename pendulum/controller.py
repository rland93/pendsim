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
    def __init__(self, init_state, pendulum, T, dt, u_max=1000, plotting=False):
        self.pendulum = pendulum 
        self.plotting = plotting
        self.setpoint = 0
        
        # System A 
        # nxn

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pendulum.g * pendulum.m)/pendulum.M, 0],
            [0, 0, 0, 1],
            [0, 0, pendulum.g/pendulum.l + pendulum.g * pendulum.m/(pendulum.l*pendulum.M), 0]
        ])

        # Input B
        # n x p

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
        constr = []
        for t in range(self.T):
            constr.append(x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t])
            constr.append(cp.abs(u[:, t]) <= self.u_max)
        cost += cp.sum_squares(x[2,self.T])
        cost += cp.sum_squares(u[0,self.T-1])
        # cost += 1e4*cp.square(x[1,0])
        constr += [x[:, 0] == state]
        problem = cp.Problem(cp.Minimize(cost), constraints=constr)
        problem.solve(verbose=True, solver='ECOS')
        action = u[0,0].value

        # dump estimate info
        self.planned_state = x[:,:].value
        self.planned_u = u[0,:].value

        print('cost: {}, u: {}, theta: {}'.format(cost.value,round(u[0,0].value,4),round(x[2,0].value,4)))
        # print('x: {}'.format(x[:,:].value))
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
        plt.pause(0.0001)


class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0

class MPCWithGPR(Controller):
    def __init__(self, window, pend, dt, plotting=False):
        self.window = window
        self.pend = pend
        self.plotting = plotting
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pend.g * pend.m)/pend.M, 0],
            [0, 0, 0, 1],
            [0, 0, pend.g/pend.l + pend.g * pend.m/(pend.l*pend.M), 0]])
        B = np.array([
            [0], 
            [1/pend.M], 
            [0], 
            [1/(pend.M * pend.l)]])
        C = np.zeros((1, A.shape[0]))
        D = np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A = sys_disc[0]
        self.B = sys_disc[1]

        self.t_priors = []
        self.x_priors = []
        self.u_priors = []
        self.tick = 0

    def policy(self, state, t, dt):
        if self.tick > self.window:

            self.t_priors.pop(0)
            self.x_priors.pop(0)
            self.u_priors.pop(0)

            epsilons = np.zeros((len(state), self.window))
            for i, (x, u) in enumerate(zip(self.x_priors, self.u_priors)):
                x = np.array(x, ndmin=2).transpose()
                if i != 0:
                    xk_true = self.x_priors[1]
                    dx = self.A @ self.x_priors[i-1]
                    xk = self.x_priors[i-1] + dt * dx
                    eps = xk_true - xk
                    epsilons[:,i] = np.array(eps)
            
            self.x_train = np.array(self.x_priors[:-1])
            self.e_train = epsilons[:,1:].transpose()
            print("x_train={}, e_train={}".format(np.shape(self.x_train),np.shape(self.e_train)))
            
            kernel = C(1) * RBF(length_scale=dt*15)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=False)
            gp.fit(self.x_train, self.e_train)

            lin_future = np.array(self.A @ state, ndmin=2)
            eps_future = gp.predict(lin_future, return_std=True)

            print("prediction={}".format(lin_future))
            print("eps_future={}".format(eps_future))

        action = 0
        self.t_priors.append(t)
        self.x_priors.append(state)
        self.u_priors.append(action)
        self.tick += 1
        return action
    def init_plot(self):
        fig = plt.figure()
        return fig
    def update_plot(self, fig):
        plt.clf()

        ax0 = fig.add_subplot(111)
        ax0.plot

        
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