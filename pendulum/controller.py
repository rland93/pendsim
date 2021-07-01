from re import X
from typing import KeysView
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import numpy as np
import cvxpy as cp
from numpy.lib.function_base import place
from scipy.optimize import zeros
from scipy.signal import cont2discrete
from scipy.signal.ltisys import StateSpaceContinuous
from pendulum.utils import array_to_kv, wrap_pi, sign
import copy
# necessary for GPR
from sklearn import preprocessing, gaussian_process
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


np.set_printoptions(precision=5,suppress=True)

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
    
    def policy(self, state, t, dt):
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
        self.window = window
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)

    def policy(self, state, t, dt):
        action = do_lqr(self.window, self.A, self.B, self.Q, self.R, state)
        data = {}
        labels = ['x', 'xd', 't', 'td']
        return action, {}

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0, {}

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
    
    def policy(self, state, t, dt):
        error = state[2] - self.setpoint
        action = 0
        if error > 0.1 and state[2] < self.threshold:
            action = -self.magnitude
        elif error < -0.1 and state[2] > -self.threshold:
            action = self.magnitude
        else:
            action = 0
        return action, {}

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
        self.window = window
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
    def policy(self, state, t, dt):
        ### Wrap 
        x = copy.deepcopy(wrap_pi(state))
        ### Solve LQR
        if np.abs(x[2]) < np.pi/4:
            action = do_lqr(self.window, self.A, self.B, self.Q, self.R, state)
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

from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.kalman import sigma_points

class LQR_UKF(Controller):
    def __init__(self, pend, dt, window, Q, R, s=10, var_window=10):
        self.pend = pend
        self.dt = dt
        self.window = window
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)
        self.s = s
        self.vw = var_window
        self.A, self.B = self._get_linear_sys(pend, dt)
        self.tick = 0
        self.priors = []
        self.kf = self.create_ukf()

    @staticmethod
    def _get_linear_sys(pend, dt):
        A, B = pend.jacA, pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A, B, C, D), dt, method='zoh')
        return sys_disc[0], np.atleast_2d(sys_disc[1])

    def create_ukf(self):
        def fx(x, dt): return self.A.dot(x)
        def hx(x): return x
        points2 = sigma_points.SimplexSigmaPoints(4)
        kf = UnscentedKalmanFilter(4, 4, self.dt, hx, fx, points2)
        # initialize noise
        kf.Q = np.diag([.2] * 4)
        # initialize smoothing
        kf.R = np.diag([self.s] * 4)
        return kf

    def policy(self, state, t, dt):
        self.priors.append(state)
        # Update variance
        l = max(self.tick - self.vw, 1)
        u = self.tick
        if self.tick >= 2:
            var = np.std(np.vstack(self.priors[l:u]), axis=0)
        else:
            var = np.asarray([0.4] * 4)
        self.kf.Q = np.diag(var)
        # Kalman filter predict
        self.kf.predict()
        # Kalman filter update
        self.kf.update(np.array(state))
        # Get control action using estimated state
        action = do_lqr(self.window, self.A, self.B, self.Q, self.R, self.kf.x)
        self.tick += 1
        # store data
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('est', labels, self.kf.x ))
        data.update(array_to_kv('var', labels, var))
        return action, data

def quadform(x, H):
    x = np.atleast_2d(x)
    if not ( (x.shape[1] == 1) or (x.shape[1] == H.shape[1]) ):
        raise ValueError('axis 1 doesn\'t match or x not a vector! x: {}, H: {}'.format(x.shape, H.shape))
    return x.T @ H @ x

def do_lqr(w, A, B, Q, R, x):
    P = [None] * (w+1)
    P[w] = Q
    for k in range(w, 0, -1):
        ApkB = A.T @ P[k] @ B
        BpkA = B.T @ P[k] @ A
        c3 = np.linalg.pinv(R + quadform(B, P[k]))
        P[k-1] = quadform(A, P[k]) - ApkB @ c3 @ BpkA + Q
    u = [None] * w
    for k in range(w):
        c1 = np.linalg.inv(R + quadform(B, P[k]))
        c2 = B.T @ P[k] @ A
        u[k] = c1 @ c2 @ x
    return float(np.squeeze(u[0]))
