import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from pendulum.utils import array_to_kv, wrap_pi, sign
import copy
# necessary for GPR
from sklearn import preprocessing, gaussian_process
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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
        state: (:obj:`float`, :obj:`float`, :obj:`float` :obj:`float`)
            The current system state
        
        Returns
        -------
        :obj:`float`
            The controller action, in force applied to the cart.
        '''
        raise NotImplementedError

class PID(Controller):
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integrator = 0
        self.prev = 0
    
    def policy(self, state, t, dt, plot=None):
        err = - (state[2]  + np.pi) % (2*np.pi) - np.pi
        errd = (err - self.prev) / dt
        self.integrator += err
        action = self.kp * err + self.ki * self.integrator + self.kd * errd
        self.prev = err

        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class LQR(Controller):
    def __init__(self, pend, dt, window, Q, R):
        self.w = window
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)

    def policy(self, state, t, dt, xref=np.array([0,0,0,0]), plot=None):
        quadform = lambda M, N: (M.T.dot(N) * M.T).sum(axis=1)
        P = [None] * (self.w+1)
        P[self.w] = self.Q

        for k in range(self.w, 0, -1):
            c1 = quadform(self.A, P[k])
            c2 = np.linalg.pinv(self.R + quadform(self.B, P[k]))
            c3 = (self.A.T @ P[k] @ self.B) @ c2 @ (self.B.T @ P[k] @ self.A)
            P[k-1] = c1 - c3

        K = [None] * self.w
        u = [None] * self.w
        for i in range(self.w):
            c1 = -np.linalg.pinv(self.R + quadform(self.B, P[k]))
            c2 = self.B.T @ P[i+1] @ self.A
            K[i] = c1 @ c2
            u[i] = K[i] @ (state - np.array([0,0,0,0]))

        action = float(np.squeeze(u[0]))
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0

class BangBang(Controller):
    def __init__(self, setpoint, magnitude, threshold):
        '''Simple "BangBang" style controller:
        if it's on turn it off
        if it's off turn it on

        Parameters
        ----------
        setpoint : :obj:`float`
            angle, radians
        magnitude : :obj:`float`
            system gain
        threshold : :obj:`float`
            max angle
        '''
        self.setpoint = setpoint
        self.magnitude = magnitude
        self.threshold = np.pi/4
    
    def policy(self, state, t, dt, plot=None):
        error = state[2] - self.setpoint
        action = 0
        if error > 0.1 and state[2] < self.threshold:
            action = -self.magnitude
        elif error < -0.1 and state[2] > -self.threshold:
            action = self.magnitude
        else:
            action = 0

        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class LQR_GPR(Controller):
    '''A controller that controls with an LQR controller (with a swing-up strategy)
    but which estimates true state using Gaussian Process Regression
    
    Parameters
    ----------
    pend : Pendulum
        The pendulum to be controlled
    dt : float
        simulation timestep
    window : int
        the number of timesteps for the LQR (forward) window
    bwindow : int
        the number of timesteps for collecting the GP train set
    Q : np.array or array-like
        4x1 array for the `Q` control matrix 
    R : float
        scalar cost of control
    
    '''
    def __init__(self, pend, dt, window, bwindow, Q, R):
        # LQR params
        self.w = window
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)
        self.pend = pend

        # model params
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])

        # GPR Params
        self.M = bwindow
        self.tick = 0
        self.priors = []

    @ignore_warnings(category=ConvergenceWarning)
    def policy(self, state, t, dt, xref=np.array([0,0,0,0]), plot=None):
        ### Wrap 
        x = copy.deepcopy(wrap_pi(state))
        ### Solve LQR
        Q = self.Q
        R = self.R
        actions = -np.squeeze(self.solve_LQR(state, Q, R))
        if np.abs(x[2]) < np.pi/4:
            action = actions[0]
        else:
            action = self.swingup(state, 50)

        if self.tick > 2:
            loweri = max(self.tick-self.M, 1)
            upperi = self.tick
            xk1 = np.atleast_2d(self.priors)[loweri:upperi] # k-1 state
            linearpred = np.dot(xk1[:,:4], self.A) + np.dot(np.atleast_2d(xk1[:,4]).T, self.B.T)
            y = np.atleast_2d(state - linearpred).T[2,:] # M x n_d
            z = np.atleast_2d(xk1) # M x n_z
            SC = preprocessing.StandardScaler()
            SC = SC.fit(z)
            z_trans = SC.transform(z)

            rq = gaussian_process.kernels.RBF(4.0, length_scale_bounds=(.5,50.0))
            ck = gaussian_process.kernels.ConstantKernel(constant_value=1.0)
            gp = gaussian_process.GaussianProcessRegressor(
                kernel=rq*ck,
                n_restarts_optimizer=10,
                alpha=1e-6
            )
            gp.fit(z_trans, y)
            indata = np.atleast_2d(list(state) + [action])
            indata_trans = SC.transform(indata)
            mu, sigma = gp.predict(indata_trans, return_std=True)
        else:
            mu, sigma = 0.0,0.0
        
        lpred = np.dot(np.atleast_2d(state), self.A) + np.dot(self.B, action).T
        nlpred = np.squeeze(lpred[0,2]) + mu

        data = {
            ('mu','t') : mu,
            ('sigma','t'): sigma,
            ('lpred','t') : np.squeeze(lpred[0,2]),
            ('nlpred','t') : nlpred
        }
        # keep track of history
        self.tick += 1
        self.priors.append(list(state) + [action])
        return action, data

    def swingup(self, x,k):
        m, g, l = self.pend.m, self.pend.g, self.pend.l
        E_norm = 2*m*g*l
        E = m * g * l * (np.cos(x[2]) - 1) # 0 = upright
        beta = E/E_norm
        u = k* beta * sign(x[3] * np.cos(x[2]))
        return - u
    
    def solve_LQR(self, state, Q, R):
        # solve an LQR policy on state with Q, R
        P = [None] * (self.w+1)
        P[self.w] = Q
        for k in range(self.w, 0, -1):
            c1 = self._quadform(self.A, P[k])
            c2 = np.linalg.pinv(R + self._quadform(self.B, P[k]))
            c3 = (self.A.T @ P[k] @ self.B) @ c2 @ (self.B.T @ P[k] @ self.A)
            P[k-1] = c1 - c3
        K = [None] * self.w
        u = [None] * self.w
        for i in range(self.w):
            c1 = -np.linalg.pinv(R + self._quadform(self.B, P[k]))
            c2 = self.B.T @ P[i+1] @ self.A
            K[i] = c1 @ c2
            u[i] = K[i] @ (np.array([0,0,0,0]) - state)
        return u
    
    @staticmethod
    def _quadform(M, N):
        return (M.T.dot(N) * M.T).sum(axis=1)s