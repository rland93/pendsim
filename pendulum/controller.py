import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

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
    def __init__(self, init_state, pendulum, T, u_max=0):
        self.pendulum = pendulum # the pendulum which we are controlling
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, (self.pendulum.g * self.pendulum.m)/self.pendulum.M, 0],
            [0, 0, 0, 1],
            [0, 0, (self.pendulum.g/self.pendulum.l + (self.pendulum.g * self.pendulum.m)/(self.pendulum.M*self.pendulum.l-2*self.pendulum.l*self.pendulum.m)),0]
           ])
        print(self.A)
        '''
        The open-loop dynamic matrix A:
        (linearized at theta = 0)

        A= [ 0, 1,                         0, 0]
           [ 0, 0,                   (g*m)/M, 0]
           [ 0, 0,                         0, 1]
           [ 0, 0, g/l + (g*m)/(M*l - 2*l*m), 0]
        '''
        self.B = np.array([[0],[1/self.pendulum.M],[0],[1/(self.pendulum.M*self.pendulum.l)]])
        '''
        The closed loop control matrix B:
        (linearized at theta = 0)

        B = [ 0     ]
            [ 1/M   ]
            [ 0     ]
            [ 1/M*l ]
        '''
        self.T = T
        self.u_max = u_max
    
    def policy(self, state, t):
        x = cp.Variable((4, self.T+1))
        u = cp.Variable((1, self.T))
        cost = 0
        constr = []
        for t in range(self.T):
            constr += [x[:,t+1] == self.A @ x[:,t] + self.B @ u[:,t]]
        # add cost of being away from theta
        cost += cp.sum_squares(x[2,:])

        constr += [
            x[:,0] == state
            ]
        
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose=False)
        print('optimal u={}'.format(u[0,0].value))
        action = u[0,0].value * 2
        # plt.plot(x[0,:].value)
        # plt.plot(x[2,:].value)
        # plt.show()
        return action

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t):
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