from scipy.signal.filter_design import normalize
from pendulum.controller import Controller
from pendulum.utils import array_to_kv, wrap_pi, sign
import numpy as np
from copy import deepcopy
from scipy.signal import cont2discrete

from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

class LQR_GPR2(Controller):
    def __init__(self, pend, dt, window, bwindow, Q):
        # LQR params
        self.w = window
        self.Q = np.diag(Q[:4])
        self.R = np.atleast_2d(Q[4])
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

    def policy(self, state, t, dt, xref=np.array([0,0,0,0]), plot=None):
        ### Wrap 
        x = deepcopy(wrap_pi(state))

        ### Solve LQR
        Q = self.Q
        R = self.R
        actions = -np.squeeze(self.solve_LQR(state, Q, R))
        if np.abs(x[2]) < np.pi/4:
            action = actions[0]
        else:
            action = self.swingup(state, 50)
        if t > 6:
            action = 0.0

        if self.tick > 2:
            loweri = max(self.tick-self.M, 1)
            upperi = self.tick
            xk1 = np.atleast_2d(self.priors)[loweri:upperi] # k-1 state
            linearpred = np.dot(xk1[:,:4], self.A) + np.dot(np.atleast_2d(xk1[:,4]).T, self.B.T)
            y = np.atleast_2d(state) - linearpred # M x n_d
            z = xk1 # M x n_z

            y = np.atleast_2d(y[:,2]).T
            z = np.atleast_2d(z[:,:])#[2,4]])

            SC = StandardScaler()
            SC = SC.fit(z)
            z_trans = SC.transform(z)

            rq = kernels.RBF(4.0, length_scale_bounds=(.5,50.0))
            ck = kernels.ConstantKernel(constant_value=1.0)
            gp = GaussianProcessRegressor(
                kernel=rq*ck,
                n_restarts_optimizer=4,
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
        return (M.T.dot(N) * M.T).sum(axis=1)

