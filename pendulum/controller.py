import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
import scipy.spatial
import matplotlib.pyplot as plt
import time
from pendulum.utils import array_to_kv
from collections import deque

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

class LQR(Controller):
    def __init__(self, pend, dt, window):
        self.w = window
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])

    def policy(self, state, t, dt, xref=np.array([0,0,0,0]), plot=None):
        quadform = lambda M, N: (M.T.dot(N) * M.T).sum(axis=1)
        Q = np.diag([0,0.5,40,0])
        R = np.atleast_2d([0.001])
        P = [None] * (self.w+1)
        P[self.w] = Q

        for k in range(self.w, 0, -1):
            c1 = quadform(self.A, P[k])
            c2 = np.linalg.pinv(R + quadform(self.B, P[k]))
            c3 = (self.A.T @ P[k] @ self.B) @ c2 @ (self.B.T @ P[k] @ self.A)
            P[k-1] = c1 - c3

        K = [None] * self.w
        u = [None] * self.w
        for i in range(self.w):
            c1 = -np.linalg.pinv(R + quadform(self.B, P[k]))
            c2 = self.B.T @ P[i+1] @ self.A
            K[i] = c1 @ c2
            u[i] = K[i] @ (state - np.array([0,0,0,0]))

        action = float(np.squeeze(u[0]))
        print('action = ' + str(action), sep='', end='\r')
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data
 
class MPC(Controller):
    def __init__(self, pend, dt, window):
        self.T = window
        self.u_max = 100
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], sys_disc[1]
        self.costW = np.diag([0, 0, 1, 0])

    def policy(self, state, t, dt, xref=np.array([0,0,0,0]), plot=None):
        constr = []
        x = cp.Variable((4, self.T + 1))
        u = cp.Variable((1, self.T))
        for t in range(self.T):
            cost = cp.quad_form(cp.abs(x[:,t+1] - xref), self.costW)
            constr += [x[:, t+1] == x[:,t] + dt * (self.A @ x[:,t] + self.B @ u[:,t])]
            constr += [x[:, 0] == state]
            constr += [cp.abs(u[0,:]) <= self.u_max]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve('ECOS', verbose=True)
        if plot is not None:
            fig, ax, lines = plot
            ax.set_xlim((np.min(list(range(self.T+1))) - 0.5, np.max(list(range(self.T+1))) + 0.1))
            for line in lines:
                if line['type'] == 'plot':
                    xdata = list(range(self.T+1))
                    ydata = np.squeeze(x[line['index'],:].value)
                    line['linesobj'].set_xdata(xdata)
                    line['linesobj'].set_ydata(ydata)

                elif line['type'] == 'hline':
                    y = xref[line['index']]
                    line['linesobj'].set_ydata([y])
                
                elif line['type'] == 'action':
                    xdata = list(range(self.T))
                    ydata = np.squeeze(u.value * 0.01)
                    line['linesobj'].set_xdata(xdata)
                    line['linesobj'].set_ydata(ydata)
            ax.set_ylim((-3, 3))
            fig.canvas.draw()
            fig.canvas.flush_events()

        action = - u[0,0].value
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0

class MPC_GPR(Controller):
    def __init__(self, pend, dt, measure_n=10, window=8):
        # prior observations
        self.M = window
        self.pend = pend
        self.measure_n = measure_n
        self.T = 10
        self.u_max = 300

        # GPR properties
        self.lenscale = 1
    
        # prior points
        self.priors = deque()

        # linear model
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (pend.g * pend.m)/pend.M, 0],
            [0, 0, 0, 1],
            [0, 0, pend.g/pend.l + pend.g * pend.m/(pend.l*pend.m), 0]])
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
        self.W = np.diag([0, 0, 4, 0.1])
        self.tick = 0

    def policy(self, state, t, dt, xref):
        data = {}
        if self.tick > self.M +1:
            # delete oldest from prior window
            self.priors.popleft()
            # first val = most recent
            xk1 = np.atleast_2d(self.priors)[:-1,:]
            xk  = np.atleast_2d(self.priors)[1:,:]
            linear_xk = np.dot(xk1[:,:4], self.A) + np.dot(np.atleast_2d(xk1[:,4]).T, self.B.T)
            y = linear_xk - xk[:,:4] # M x n_d
            z = xk1 # M x n_z
            # create cov matrix
            L = self.create_prior_matr(z, y, self.lenscale)
            
            x = cp.Variable((4, self.T+1))
            # ctrl act
            u = cp.Variable((1, self.T))
            # mu
            mu = []
            constr = []
            cost = 0
            for t in range(self.T):
                # for each output dimension
                for a in range(y.shape[1]):
                    
                    # calcuate alpha
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y[:,a]))
                    # find z star by applying kernel on z_new and z
                    # where z_new is x[t,:] (the state at future time t)

                    #diff = spatial.distance.cdist(z_new, z, metric='sqeuclidean')
                    # diff = 
                    # for rowi in z_new == 1
                    #     for rowj in z == 8,
                    #         find sqeuclid dist between 2 rows
                    #     endfor
                    # endfor
                    # diff is a rowi by rowj matr of scalar distances
                    
                    # for example
                    # z_new is 1x4
                    # z is 5x4
                    # diff is now
                    # 1x5

                    # diff = np.exp(diff/lenscale * - 0.5)
                    # mu[0,a] = z_star.dot(alpha)
                    # mean
                    '''
                    print('\talpha: {}'.format(alpha.shape))
                    print('\tz[:, a]: {} = m x 1 (column vector)'.format(z[:, a].shape))
                    print('\tx[a, t]: {} = scalar'.format(x[a, t].shape))
                    print('\tz[:, a] - x[a, t]: {}'.format((z[:, a] - x[a, t]).shape))
                    print('\tabs(z[:, a] - x[a, t]): {}'.format(cp.abs(z[:, a] - x[a, t]).shape))
                    print(
                        '\texp(abs(z[:, a] - x[a, t])): {}'.format(
                        (cp.exp(cp.abs(z[:, a] - x[a, t])) @ alpha.T).shape
                        )
                    )
                    print(
                        '\tself.A * x[a,t]: {}'.format(
                        (np.linalg.norm(self.A[:,a], ord=1) * x[a, t]).shape
                        )
                    )
                    constr.append(
                        x[a, t + 1] == np.sum(self.A[:,a]) * x[a,t] + \
                            np.sum(self.B[a, 0]) * u[0, t]
                        # cp.exp(cp.abs(z[:, a] - x[a, t])) @ alpha.T 
                    )
                    '''
                # system model constraint
                constr.append(
                    x[:, t+1] == x[:,t] + \
                        self.A[:,a] @ x[:,t] + \
                        self.B @ u[:,t]
                )
                # max control action
                constr += [cp.abs(u[:, t]) <= self.u_max]
                cost += cp.quad_form(x[:,t+1] - xref, self.W)
                cost += cp.sum_squares(u[:,t])
            constr += [x[:, 0] == state]
            problem = cp.Problem(cp.Minimize(cost), constraints=constr)
            problem.solve(verbose=False)
            print('cost: {}'.format(cost.value))
            action = u[0,0].value


            data = {}
            labels = ['x', 'xd', 't', 'td']
            data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
            '''
            mu, sig = self.make_prediction(L, self.lenscale, z, y, state, self.prior_action)
            # make linear, nonlinear predictions
            lpred = np.dot(np.atleast_2d(state), self.A)
            nlpred = lpred - mu

            # make n-ahead linear, nonlinear predictions
            lpred_n = np.atleast_2d(state)
            nlpred_n = np.atleast_2d(state)
            for _ in range(self.measure_n):
                lpred_n = np.dot(lpred_n, self.A)
                mu, sig = self.make_prediction(L, self.lenscale, z, y, nlpred_n, self.prior_action)
                nlpred_n = np.dot(nlpred_n, self.A) - mu

            # write data
            labels = ['x', 'xd', 't', 'td']
            level1_keys = ['mu', 'sigma', 'lpred', 'nlpred', 'lpred_n', 'nlpred_n']
            values = [mu, sig, lpred, nlpred, lpred_n, nlpred_n]
            data = {}
            for l1, val in zip(level1_keys, values):
                data.update(pendulum.array_to_kv(l1, labels, np.squeeze(val)))
            '''
        else:
            '''
            # write data
            labels = ['x', 'xd', 't', 'td']
            level1_keys = ['mu', 'sigma', 'lpred', 'nlpred', 'lpred_n', 'nlpred_n']
            values = [np.empty(state.shape[0]) for k in level1_keys]
            data = {}
            for l1, val in zip(level1_keys, values):
                data.update(pendulum.array_to_kv(l1, labels, val))
            '''
            data = {}
            labels = ['x', 'xd', 't', 'td']
            data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
            print(data)
            action = 0

        # no action for now
        # build prior window
        self.prior_action = action
        self.priors.append(list(state) + [action])
        # we increment tick every time we take ctrl action
        self.tick += 1
        return action, data
    
    def apply_kernel(self, x1, x2=None, lenscale=1):
        if x2 is None:
            diff = spatial.distance.pdist(x1, metric='sqeuclidean')
            gram = np.exp(diff/lenscale * -0.5)
            gram = spatial.distance.squareform(gram)
            np.fill_diagonal(gram, 1)
        else:
            diff = spatial.distance.cdist(x1, x2, metric='sqeuclidean')
            gram = np.exp(diff/lenscale * - 0.5)
        return gram

    def ll_loss(self, z, y, theta):
        K = self.apply_kernel(z, a=theta)
        K[np.diag_indices_from(K)] += np.var(y, axis=1) + 1e-8
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        a = -0.5 * np.dot(y.T, alpha)
        b = -0.5 * np.log(np.trace(L))
        c = -0.5 * L.shape[0] * np.log(np.pi * 2)
        ll = a + b + c
        return ll.sum()
    
    def optimize(self, z, y):
        theta = 1
        def obj_func(theta, z, y):
            return -self.ll_loss(z, y, theta)

        results = optimize.minimize(
            obj_func,
            theta,
            (z, y),
            method='L-BFGS-B',
            options={'disp': True}
        )
        print('\ttheta={}'.format(results['x']))
        return(results['x'])

    def create_prior_matr(self, z, y, l):
        '''
        Return cholesky of cov matrix K built from 
        z, y, and l
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

        
class BangBang(Controller):
    def __init__(self, setpoint, magnitude):
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
        self.set_theta = set_theta
        self.magnitude = magnitude
        self.threshold = threshold
    
    def policy(self, state, t):
        error = state[2] - self.setpoint
        if error > 0.1 and state[2] < self.threshold:
            return self.magnitude
        elif error < -0.1 and state[2] > -self.threshold:
            return -self.magnitude
        else:
            return 0