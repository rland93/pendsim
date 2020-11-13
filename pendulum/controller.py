import numpy as np
from scipy import spatial
import cvxpy as cp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process
import time

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
            [1/pendulum.M*pendulum.l]
            
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
        cost += 1e4*cp.sum_squares(u[0,self.T-1])
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
    def __init__(self, window, pend, dt, every, plotting=False):
        # prior observations
        self.M = window

        self.pend = pend
        self.plotting = plotting

        self.l_pred_x_k = np.zeros(4)
        self.l_err_x_k = np.zeros(4)
        self.nl_pred_x_k = np.zeros(4)
        self.nl_err_x_k = np.zeros(4)

        self.pred_mu = np.zeros(4)
        self.pred_sig = np.zeros(4)

        self.l_pred_state = np.zeros(4)
        self.nl_pred_state = np.zeros(4)
    
        self.have_pred = False

        self.z_points = np.zeros((self.M,5))
        self.y_points = np.zeros((self.M,4))

        self.z_current = np.zeros(5)
        

        # alter parameters to produce a (very) inaccurate linearized model
        newL = pend.l# + np.random.random_sample()
        newm = pend.m# + np.random.random_sample()
        newM = pend.M# + np.random.random_sample()
        newg = pend.g# + np.random.random_sample()

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (newg * newm)/newM, 0],
            [0, 0, 0, 1],
            [0, 0, newg/newL + newg * newm/(newL*newM), 0]])
        B = np.array([
            [0], 
            [1/pend.M], 
            [0], 
            [1/(pend.M * pend.l)]])
        C = np.zeros((1, A.shape[0]))
        D = np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt * every, method='zoh')
        self.A = sys_disc[0]
        self.B = sys_disc[1]

        self.t_priors = []
        self.x_priors = []
        self.u_priors = []
        self.tick = 0

    def apply_kernel(self, x1, x2=None, a=1):
        if x2 is None:
            diff = spatial.distance.pdist(x1/a, metric='sqeuclidean')
            gram = np.exp(diff/2 * -1)
            gram = spatial.distance.squareform(gram)
            np.fill_diagonal(gram, 1)
        else:
            diff = spatial.distance.cdist(x1/a, x2/a, metric='sqeuclidean')
            gram = np.exp(diff/2 * -1)
        return gram

    def make_prediction(self, z, y, z_new):
        n_d = np.shape(y)[1]

        mu = np.zeros(n_d)
        sigma = np.zeros(n_d)
        K = self.apply_kernel(z)

        for a in range(n_d):
            # add noise
            K_a = K + np.eye(np.shape(K)[0]) * np.var(y[:,a])
            L = np.linalg.cholesky(K_a)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y[:,a]))
            
            print(np.shape(y[:,a]))
            print(np.shape(z))

            dist = self.apply_kernel(z, np.atleast_2d(y[:,a]))
            mu[a] = dist.dot(alpha)
        return mu

    def train(self, z, y):
        M = np.shape(z)[0]
        n_d = np.shape(y)[1]
        n_z = np.shape(z)[1]
        KZZ = np.zeros((n_d, M, M))
        sigma_y = [np.var(y[:,i]) for i in range(n_d)]
        for a in range(n_d):
            KZZ[a,:,:] = self.apply_kernel(z)
        return KZZ

    def predict(self, z, y, KZZ, z_new):
        M = np.shape(z)[0]
        n_d = np.shape(y)[1]
        n_z = np.shape(z)[1]
        KzZ = np.zeros((n_d, M))
        KZz = np.zeros((M, n_d))
        Kzz = np.zeros((n_d))
        sigma_y = [np.var(y[:,i]) for i in range(n_d)]
        mu = np.zeros((n_d))
        sg = np.zeros((n_d))

        for a in range(n_d):
            # solve KZz etc with z_new
            KZz[:,a] = sigma_y[a] * sigma_y[a] * np.atleast_2d([self.apply_kernel(z[i,:], np.ravel(z_new)) for i in range(M)])
            KzZ[a,:] = np.transpose(KZz[:,a])
            Kzz[a] = sigma_y[a] * sigma_y[a] * self.apply_kernel(np.ravel(z_new), np.ravel(z_new))
            # Solve mu, sigma
            invs = KzZ[a,:] @ np.linalg.inv(KZZ[a,:,:] + np.eye(M) * sigma_y[a] * sigma_y[a])
            mu[a] = invs @ y[:,a]
            sg[a] = Kzz[a] - invs @ KZz[:,a]
        return mu, sg

    def policy(self, state, t, dt):
        print('=================================================')        
        if self.tick > self.M:
            t0 = time.time()

            if self.have_pred:
                # measure prediction errors
                self.l_pred_x_k = self.l_pred_state
                self.l_err_x_k = np.abs(state - self.l_pred_state)
                self.nl_pred_x_k = self.nl_pred_state
                self.nl_err_x_k = np.abs(state - self.nl_pred_state)
            # remove oldest states
            self.t_priors.pop(0)
            self.x_priors.pop(0)
            self.u_priors.pop(0)

            # training set D {y, z}
            z = np.zeros((self.M, len(state) + 1))
            y = np.zeros((self.M, len(state)))

            # x_k-1
            xk1 = np.zeros((self.M, len(state) + 1))
            xk1[:, :4] = np.flip(self.x_priors, axis=0)
            xk1[:,  4] = np.flip(self.u_priors, axis=0)
            
            # x_k
            xk = np.zeros((self.M, len(state)))
            xk[0, :] = state
            xk[1:, :] = np.flip(self.x_priors, axis=0)[:-1,:]
            
            # Output y is the error between linear model 
            # forecast for x_k (from state at x_k-1) and 
            # reality at x_k.
            linear_xk = np.dot(np.atleast_2d(xk1[:,:4]), self.A) + np.dot(np.atleast_2d(xk1[:, 4]).T, self.B.T)
            y = linear_xk - xk
            z[:, :] = xk1[:, :]
            
            self.z_current = np.zeros(len(state) + 1)
            self.z_current[:4] = np.atleast_2d(state)
            self.z_current[4] = np.atleast_2d(0)

            # train
            KZZ = self.train(z,y)
           
            # make predictions
            self.l_pred_state = np.zeros(4)
            self.nl_pred_state = np.zeros(4)

            # linear pred
            self.l_pred_state = self.A @ self.z_current[:4]
            # nonlinear pred
            self.pred_mu = self.make_prediction(z, y, self.z_current)
            # self.pred_mu, self.pred_sg = self.predict(z, y, KZZ, self.z_current)
            self.nl_pred_state = self.pred_mu + self.l_pred_state
            print('\tmu: {}'.format(self.pred_mu))
            # print('\tmu: {}\n\tsigma: {}'.format(self.pred_mu, self.pred_sig))
            
            self.z_points = z
            self.y_points = y

            self.have_pred = True
            t1 = time.time()
            print('executed in '.format(t1-t0))

            action = 0
        else:
            action = 0
        

        self.tick += 1
        self.t_priors.append(t)
        self.x_priors.append(state)
        self.u_priors.append(action)
    
        return action

    def init_plot(self):
        fig = plt.figure()
        return fig
    def update_plot(self, fig):
        plt.clf()
        labels = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
        ax0 = fig.add_subplot(221)
        ax0.scatter(self.z_points[:, 0], self.y_points[:, 0], c='k', label='training set')
        ax0.scatter(self.z_current[0], self.pred_mu[0], c='r', label='prediction')
        ax0.set_xlabel(labels[0])
        ax0.set_ylabel('error in ' + labels[0])
        ax0.set_title('predicting ' + labels[0])

        ax1 = fig.add_subplot(222)
        ax1.scatter(self.z_points[:, 1], self.y_points[:, 1], c='k', label='training set')
        ax1.scatter(self.z_current[1], self.pred_mu[1], c='r', label='prediction')
        ax1.set_xlabel(labels[1])
        ax1.set_ylabel('error in ' + labels[1])
        ax1.set_title('predicting ' + labels[1])

        ax2 = fig.add_subplot(223)
        ax2.scatter(self.z_points[:, 2], self.y_points[:, 2], c='k', label='training set')
        ax2.scatter(self.z_current[2], self.pred_mu[2], c='r', label='prediction')
        ax2.set_xlabel(labels[2])
        ax2.set_ylabel('error in ' + labels[2])
        ax2.set_title('predicting ' + labels[2])

        ax3 = fig.add_subplot(224)
        ax3.scatter(self.z_points[:, 3], self.y_points[:, 3], c='k', label='training set')
        ax3.scatter(self.z_current[3], self.pred_mu[3], c='r', label='prediction')
        ax3.set_xlabel(labels[3])
        ax3.set_ylabel('error in ' + labels[3])
        ax3.set_title('predicting ' + labels[3])

        plt.draw()
        plt.pause(0.001)
        
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