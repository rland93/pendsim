import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process

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

    def square_exp(self, z_i, z_j, sigma=1, length=1):
        '''
        squared exponential kernel
        '''
        L = np.eye(np.shape(z_i)[0])
        l = np.transpose(z_i - z_j) @ L @ (z_i - z_j)
        e = sigma * sigma * np.exp(-0.5 * l)
        return e

    def compute_gram(self, z, sigma=1):

        a = 2
        n = np.shape(z)[0]
        gram = np.zeros([n, n])
        # lower triangular
        for i in range(n):
            for j in range(i + 1):
                    r = np.linalg.norm(z[i,:] - z[j,:]) / a
                    gram[i, j] = np.exp(- r * r)

        # upper triangular
        upper_tri_idx = np.triu_indices(n)
        gram[upper_tri_idx] = gram.T[upper_tri_idx]

        # plt.matshow(gram)
        # plt.show()
        return gram

    def train(self, z, y):
        M = np.shape(z)[0]
        n_d = np.shape(y)[1]
        n_z = np.shape(z)[1]
        KZZ = np.zeros((n_d, M, M))
        sigma_y = [np.var(y[:,i]) for i in range(n_d)]
        for a in range(n_d):
            KZZ[a,:,:] = self.compute_gram(z, sigma=sigma_y[a])
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
            KZz[:,a] = sigma_y[a] * sigma_y[a] * np.atleast_2d([self.square_exp(z[i,:], np.ravel(z_new), sigma_y[a]) for i in range(M)])
            KzZ[a,:] = np.transpose(KZz[:,a])
            Kzz[a] = sigma_y[a] * sigma_y[a] * self.square_exp(np.ravel(z_new), np.ravel(z_new), sigma_y[a])
            # Solve mu, sigma
            invs = KzZ[a,:] @ np.linalg.inv(KZZ[a,:,:] + np.eye(M) * sigma_y[a] * sigma_y[a])
            mu[a] = invs @ y[:,a]
            sg[a] = Kzz[a] - invs @ KZz[:,a]
        return mu, sg

    def policy(self, state, t, dt):
        print('=================================================')        
        if self.tick > self.M + 1:
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
            y = np.zeros((self.M, len(state)))
            z = np.zeros((self.M, len(state) + 1))

            # build train set D {y, z}
            for i in range(self.M):
                xj1_true = self.x_priors[i+1] # we keep the window at size M+1 so we can do this.
                fxj = self.A @ self.x_priors[i]
                # y
                y[i,:] = xj1_true - fxj
                # z
                z[i,:] = np.concatenate((self.x_priors[i].transpose(), [self.u_priors[i]]))

            avgy = np.average(y, axis=0)
            # train
            KZZ = self.train(z,y)
           
            # make prediction on a new point z_new
            z_new = np.zeros(5)
            z_new[:4] = np.atleast_2d(state)
            z_new[4] = 0
            
            # prediction is predicted deviation from linear model
            self.pred_mu, self.pred_sig = self.predict(z, y, KZZ, z_new)
            print('\tmean training y value: {}'.format(avgy))            
            print('\tmu: shape {}, value={}\n\tsigma: shape {}, value={}'.format(np.shape(self.pred_mu), self.pred_mu, np.shape(self.pred_sig), self.pred_sig))

            # linear predicted k+1 state
            self.l_pred_state = self.A @ state 
            # non-linear predicted k+1 state
            self.nl_pred_state = self.A @ state + self.pred_mu

            self.have_pred = True

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
        ax0 = fig.add_subplot(111)
        plt.draw()
        plt.pause(0.0001)
        
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