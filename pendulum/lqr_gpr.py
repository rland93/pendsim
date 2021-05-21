from numpy.core.defchararray import lower
from pendulum.controller import Controller
from pendulum.utils import array_to_kv, wrap_pi, sign
import numpy as np
from copy import deepcopy
from collections import deque
from scipy.signal import cont2discrete
from scipy import spatial
from time import sleep

class LQR_GPR(Controller):
    def __init__(self, pend, dt, window, bwindow, Q):
        # LQR params
        self.w = window
        self.Q = np.diag(Q[:4])
        self.R = np.atleast_2d(Q[4])
        self.pend = pend

        self.priorstate = []

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

    @staticmethod
    def _quadform(M, N):
        return (M.T.dot(N) * M.T).sum(axis=1)

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

        if self.tick > 1:
            loweri = max(self.tick-self.M, 1)
            upperi = self.tick
            xk1 = np.atleast_2d(self.priors)[loweri:upperi] # k-1 state

            #print(xk1[-1], '\n', np.array(self.priorstate), '\n')

            linearpred = np.dot(xk1[:,:4], self.A) + np.dot(np.atleast_2d(xk1[:,4]).T, self.B.T)
            y = np.atleast_2d(state) - linearpred # M x n_d

            train = y[-1]
            
            z = xk1 # M x n_z
            L = self.create_prior_matr(z, y, 1)
            mu, sigma = self.make_prediction(L, 1, z, y, state, action)


            if plot is not None:
                fig, ax, sc, line1, line2 = plot
                xs = list(range(y.shape[0] + 1))
                ys = y[:,2].tolist() + mu[:,2].tolist()
                ys_upper = list(y[:,2] + y[:,2].var())
                ys_lower = list(y[:,2] - y[:,2].var())
                sc.set_offsets(np.column_stack((xs, ys)))
                ax.set_xlim( min(xs), max(xs) )
                ax.set_ylim( min(min(ys),0), max(max(ys),0) )

                for l1, l2 in zip(line1,line2):
                    l1.set_data(xs[:-1], ys_upper)
                    l2.set_data(xs[:-1], ys_lower)


                fig.canvas.draw()
                fig.canvas.flush_events()



            
        else:
            train = np.zeros( (1, 4), dtype=np.float64)
            mu, sigma = np.zeros( (1, 2) , dtype=np.float64), np.zeros( (1, 2), dtype=np.float64)
        
        lpred = np.dot(np.atleast_2d(state), self.A) + np.dot(self.B, action).T
        nlpred = lpred - mu

        data = {
            ('mu','t') : mu
        }
        labels = ['x', 'xd', 't', 'td']
        # data.update(array_to_kv('train', labels, np.squeeze(train)))
        # data.update(array_to_kv('mu', labels, np.squeeze(mu)))
        data.update(array_to_kv('sigma', labels, np.squeeze(sigma)))
        data.update(array_to_kv('lpred', labels, np.squeeze(lpred)))
        data.update(array_to_kv('nlpred', labels, np.squeeze(nlpred)))
        
        # keep track of history
        self.tick += 1
        self.priors.append(list(state) + [action])
        ### Test prior state
        self.priorstate = list(state) + [action]
        return action, data

    def create_prior_matr(self, z, y, l):
        '''
        Return cholesky of cov matrix K built from 
        z, y, and lr
        '''
        # length scale
        K = self.apply_kernel(z, lenscale=l)
        K[np.diag_indices_from(K)] += np.var(y, axis=1) + 1e-9
        L = np.linalg.cholesky(K)
        return L

    def make_prediction(self, L, l, z, y, x, u):
        '''
        L: cholesky of cov matrix K
        l: length parameter of matrix
        z: prior   x, u
        y: prior g(x, u)

        x: state
        u: control input
        '''
        # prefill
        mu = np.zeros((1,np.shape(y)[1]))
        sigma = np.zeros((1,np.shape(y)[1]))
        z_new = np.empty((1,5))
        z_new[:, :4] = np.asarray(x)
        z_new[:, 4] = np.asarray(u)
        # we train one model for each dim of the output y.
        for a in range(y.shape[1]):
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y[:,a]))
            # mean
            z_star = self.apply_kernel(z_new, z, lenscale=l)
            mu[0,a] = z_star.dot(alpha)
            # variance
            v = np.linalg.solve(L, z_star.T)
            sigma[0,a] = self.apply_kernel(z_new, z_new, lenscale=l) - np.dot(v.T, v)
        # sigma = np.sqrt(sigma)
        return mu, sigma
    
    def apply_kernel(self, x1, x2=None, lenscale=1):
        if x2 is None:
            diff = spatial.distance.pdist(x1, metric='sqeuclidean')
            gram = np.exp(diff/lenscale * -0.5)
            gram = spatial.distance.squareform(gram)
            np.fill_diagonal(gram, 1)
        else:
            # sqeuclidean
            diff = spatial.distance.cdist(x1, x2, metric='sqeuclidean')
            gram = np.exp(diff/lenscale * - 0.5)
        return gram
    
    def swingup(self, x,k):
        m, g, l = self.pend.m, self.pend.g, self.pend.l
        E_norm = 2*m*g*l
        E = m * g * l * (np.cos(x[2]) - 1) # 0 = upright
        beta = E/E_norm
        u = k* beta * sign(x[3] * np.cos(x[2]))
        return - u
