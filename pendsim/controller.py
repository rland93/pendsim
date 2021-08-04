import pendsim.sim, pendsim.utils
import copy

import numpy as np
from scipy.signal import cont2discrete

# needed for LQR + UKF
from filterpy.kalman import UnscentedKalmanFilter, sigma_points

# needed for typeannotations
from typing import Tuple

LABELS = ["x", "xd", "t", "td"]


class Controller(object):
    """
    Base class for controllers. A controller executes a `policy` during
    the simulation loop. It takes some measured `state` and takes some
    `action` on that state. The `action` is a force applied to the base
    of the cart.

    A controller's `policy` can include a state estimator or functions
    we can use to record data.

    Methods
    -------
    policy:
        control policy

    do_swingup:
        get action from Astrom's "swing-up" policy

    do_lqr:
        get action from an LQR policy

    get_linear_sys:
        get the linearized system from jacobians

    store_4tuple:
        store a 4-tuple into multi-index

    wrapPi:
        wrap an angular value to [-pi, pi] interval

    create_ukf:
        create an unscented kalman filter for the system

    do_pid:
        get action from a PID control policy

    get_and_store_priors:
        utility function for storing state priors in a moving backwards horizon

    """

    def __init__(self, init_state: np.ndarray):
        """init function for Controller object. Rare that this would be used
        directly.

        Parameters
        ----------
        init_state : np.ndarray
            Initial state.
        """
        self.init_state = init_state
        self.prev_err = 0

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        raise NotImplementedError

    def do_swingup(
        self, pend: pendsim.sim.Pendulum, state: np.ndarray, k: float
    ) -> float:
        """Implement a swing-up by Energy control method described by Astrom
        and Furuta. Typically, we want to implement the swing-up strategy if
        the pendulum is below some threshold (i.e. theta > pi/2).

        See

        Åström, Karl Johan, and Katsuhisa Furuta. "Swinging up a pendulum by
        energy control." Automatica 36.2 (2000): 287-295.
        at https://doi.org/10.1016/S1474-6670(17)57951-3

        Parameters
        ----------
        pend : pendsim.sim.Pendulum
            The pendulum object containing `m`,`g`,`l` parameters we use to
            estimate the energy of the system.
        state : np.ndarray
            The current state
        k : float
            Swing-up gain

        Returns
        -------
        float
            controller action.
        """
        m, g, l = pend.m, pend.g, pend.l
        E_norm = 2 * m * g * l
        E = m * g * l * (np.cos(state[2]) - 1)  # 0 = upright
        beta = E / E_norm
        u = k * beta * pendsim.utils.sign(state[3] * np.cos(state[2]))
        return -u

    def do_lqr(
        self,
        w: int,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Finite-horizon discrete Linear Quadratic Regulator policy.
        An LQR controller produces an optimal control policy over a finite horizon
        according to a quadratic cost over the system state.

        See https://underactuated.mit.edu/lqr.html


        Parameters
        ----------
        w : int
            window over which to perform LQR control
        A : np.ndarray
            linear plant transition matrix
        B : np.ndarray
            linear control matrix
        Q : np.ndarray
            controller gain matrix
        R : np.ndarray
            control action penalty
        x : np.ndarray
            system state

        Returns
        -------
        np.ndarray
            sequence of control actions
        """

        P = [None] * (w + 1)
        P[w] = Q
        for k in range(w, 0, -1):
            p1 = A.T @ P[k] @ A  # (4,4)
            p2 = A.T @ P[k] @ B  # (4,1)
            p3 = R + B.T @ P[k] @ B  # (4,1)
            p3 = np.linalg.pinv(R + B.T @ P[k] @ B)
            p4 = B.T @ P[k] @ A
            P[k - 1] = p1 - p2 @ p3 @ p4 + Q

        u = [None] * w
        for k in range(w):
            c1 = np.linalg.inv(R + B.T @ P[k] @ B)
            c2 = B.T @ P[k] @ A
            u[k] = c1 @ c2 @ x
        return np.squeeze(u)

    def get_linear_sys(
        self, Adot: np.ndarray, Bdot: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearize from jacobians of system with timestep `dt`.


        Parameters
        ----------
        Adot : np.ndarray
            The (4,4) jacobian of the plant matrix A
        Bdot : np.ndarray
            The (4,1) jacobian of the control matrix B
        dt : float
            control timestep

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            linearized system matrices A, B.
        """
        A, B = Adot, Bdot
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A, B, C, D), dt, method="zoh")
        return sys_disc[0], sys_disc[1]

    def store_4tuple(self, level1_key: str, val: np.ndarray) -> dict:
        """Helper function for storing a 4-tuple of values. A level 0 key
        is given, and the level 1 keys are populated with 'x', 'xd', 't','td'
        values. Use this when you need to easily return data from a 4-tuple
        produced by the simulation that maps to each of the 4 state values.

        Example

        >>> val = [1,2,3,4]
        >>> key = "count"
        >>> store4tuple(key, val)
        {
            ('count','x') : 1,
            ('count','xd') : 2,
            ('count','t') : 3,
            ('count','td') : 4,
        }

        Parameters
        ----------
        level1_key : str
            outer key
        val : np.ndarray
            values. Must be shape (4,1) or (4,)

        Returns
        -------
        dict
            dict containing new data values
        """
        labels = ["x", "xd", "t", "td"]
        return pendsim.utils.array_to_kv(level1_key, labels, np.squeeze(val))

    def wrapPi(self, val: float) -> float:
        """Wrap an angle to the interval between [-pi, pi].

        Parameters
        ----------
        val : float
            angle

        Returns
        -------
        float
            wrapped angle
        """
        return (val + np.pi) % (2 * np.pi) - np.pi

    def create_ukf(
        self, dt: float, hx: callable, fx: callable
    ) -> UnscentedKalmanFilter:
        """Create an unscented Kalman filter with state transition functions
        `fx`, measurement function `hx` with timestep between measurements
        estimated as `dt`.

        Parameters
        ----------
        dt : float
            control timestep
        hx : callable
            function describing mapping between sensor inputs and measurements
        fx : callable
            function describing system. Can be non-linear.

        Returns
        -------
        UnscentedKalmanFilter
            UKF object with sensible defaults for sigma points.
        """
        n = 4
        points = sigma_points.MerweScaledSigmaPoints(
            n,
            alpha=1e-4,
            beta=2,
            kappa=3 - n,
        )

        kf = UnscentedKalmanFilter(
            dim_x=n,
            dim_z=n,
            dt=dt,
            hx=hx,
            fx=fx,
            points=points,
        )
        return kf

    def do_pid(
        self, dt: float, kp: float, ki: float, kd: float, state: np.array
    ) -> float:
        """Perform PID (proportional-integral-derivative) control policy given
        a system state. This policy executes on the pendulum angle (i.e. attempts)
        to keep the pendulum in the upright position.

        Parameters
        ----------
        dt : float
            control timestep
        kp : float
            proportional gain
        ki : float
            integral gain
        kd : float
            derivative gain
        state : np.array
            input state

        Returns
        -------
        float
            control action
        """
        err = -self.wrapPi(state[2])
        errd = (err - self.prev_err) / dt
        self.integrator += err
        action = kp * err + ki * self.integrator + kd * errd
        self.prev_err = err
        return action

    def get_and_store_priors(self, state: np.ndarray, n: int) -> Tuple[int, int]:
        """utility function for getting and storing priors in some window `n`.
        This function returns two values `l` and `u` which provide the indices
        to a moving window over prior_states. You must include an attr `tick`,
        which is incremented to determine the window, as well as an attr `prior_states`
        to store states to use this function, and this function alters those attrs.

        Parameters
        ----------
        state : np.ndarray
            input state to store
        n : int
            window for returning `l`, `u` pair.

        Returns
        -------
        Tuple[int, int]
            l, u pair. If you want to get n most recent prior measurements, call
            self.prior_states[l:u]
        """
        l = max(0, self.tick - n)
        u = self.tick
        self.tick += 1
        self.prior_states.append(state)
        return l, u


class PID(Controller):
    """Basic PID Control class. This class produces a basic PID controller
    that takes the state as it is measured and executes a pid policy on it.

    Parameters
    ----------
    pid : Tuple[float, float, float]
        kp, ki, kd gains for the controller
    """

    def __init__(self, pid: Tuple[float, float, float]) -> None:
        self.kp, self.ki, self.kd = pid
        self.integrator = 0
        self.prev_err = 0

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        """PID policy. This just calls `do_pid` on the state, but you can
        add data collection values to it.

        Parameters
        ----------
        state : np.ndarray
            system state
        dt : float
            control timestep

        Returns
        -------
        Tuple[float, dict]
            action, data pair
        """
        action = self.do_pid(dt, self.kp, self.ki, self.kd, state)
        return action, {}


class PID_UKF(Controller):
    """PID controller with unscented kalman filter. The unscented kalman filter
    uses variance on the scale of `var_t` to get the measurement noise estimate.
    (`var_t` is a measurement of time in seconds.)


    Parameters
    ----------
    pid : Tuple[float, float, float]
        kp, ki, kd gains for the controller
    pend : [type]
        object containing the pendulum under control.
    dt : float
        control timestep
    var_t : float
        window over which to collect the measurement
    """

    def __init__(
        self, pid: Tuple[float, float, float], pend, dt: float, var_t: float
    ) -> None:
        # PID Parameters
        self.kp, self.ki, self.kd = pid
        self.integrator = 0
        self.prev_err = 0
        # UKF Parameters
        self.var_t = var_t
        self.A, self.B = self.get_linear_sys(pend.jacA, pend.jacB, dt)

        def fx(x, dt, u=0):
            return np.dot(self.A, x) + np.dot(self.B, u).T

        def hx(x):
            return x

        self.kf = self.create_ukf(dt, hx, fx)

        self.prior_states = []
        self.tick = 0

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        """Measure variance from prior states over `var_t`, then use that variance
        to compute an estimated state with an unscented kalman filter. The measured
        state is then used for the PID control policy.

        Parameters
        ----------
        state : np.ndarray
            system state
        dt : float
            timestep

        Returns
        -------
        Tuple[float, dict]
            action, data pair
        """
        self.get_and_store_priors(state, round(self.var_t * 1 / dt))
        prior = np.array(self.prior_states)
        if self.tick > 1:
            var = np.cov(prior.T)
        else:
            var = np.eye(4) * 1

        self.kf.Q = var ** 2
        self.kf.update(state)
        est = self.kf.x
        res = self.kf.y

        action = self.do_pid(dt, self.kp, self.ki, self.kd, state)
        self.kf.predict(dt, **{"u": action})

        labels = ["x", "xd", "t", "td"]
        data = {}
        data.update(pendsim.utils.array_to_kv("res", labels, res))
        data.update(pendsim.utils.array_to_kv("est", labels, est))
        return action, data


class LQR_UKF(Controller):
    """LQR controller with unscented kalman filter state estimation.

    Parameters
    ----------
    qr : Tuple[np.ndarray, float]
        tuple of (Q, R) matrices. In this case R is a 1x1 matrix or float.
    lqrw : int
        the window over which to perform LQR. Longer windows are more accurate,
        because they take into account a longer approximation of the system
        response. However, they take more time to compute and can be numerically
        unstable.
    pend : pendsim.sim.Pendulum
        pendulum object with jacobians of A and B matrices.
    dt : float
        control timestep
    var_t : float
        window over which to collect variances, in units of seconds.
    """

    def __init__(
        self, qr: Tuple[np.ndarray, float], lqrw: int, pend, dt: float, var_t: float
    ) -> None:
        # LQR Parameters
        self.Q, self.R = qr
        self.lqrw = lqrw
        # UKF Parameters
        self.A, self.B = self.get_linear_sys(pend.jacA, pend.jacB, dt)

        def fx(x, dt, u=0):
            return np.dot(self.A, x) + np.dot(self.B, u).T

        def hx(x):
            return x

        self.kf = self.create_ukf(dt, hx, fx)

        # Prior state info
        self.tick = 0
        self.var_t = var_t
        self.prior_states = []

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        self.get_and_store_priors(state, round(self.var_t * 1 / dt))
        prior = np.array(self.prior_states)
        if self.tick > 5:
            var = np.cov(prior.T)
        else:
            var = np.eye(4) * 1

        self.kf.Q = var ** 2
        self.kf.update(state)
        est = self.kf.x
        res = self.kf.y

        action = self.do_lqr(self.lqrw, self.A, self.B, self.Q, self.R, est)
        self.kf.predict(dt, **{"u": action})

        labels = ["x", "xd", "t", "td"]
        data = {}
        data.update(pendsim.utils.array_to_kv("res", labels, res))
        data.update(pendsim.utils.array_to_kv("est", labels, est))
        return action, data


class LQR(Controller):
    """Perform an LQR strategy.

    Parameters
    ----------
    pend : pendsim.sim.Pendulum
        The pendulum we want to control
    dt : float
        Timestep of the simulation
    window : int
        the window over which to perform LQR. For example; `window=5` will
        optimize over 5 timesteps
    Q : np.ndarray
        State cost array
    R : np.ndarray
        Actuation cost array
    """

    def __init__(
        self,
        pend: pendsim.sim.Pendulum,
        dt: float,
        window: int,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.window = window
        self.A, self.B = self.get_linear_sys(pend.jacA, pend.jacB, dt)
        self.Q = Q
        self.R = R

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        action = self.do_lqr(self.window, self.A, self.B, self.Q, self.R, state)
        return action, {}


class LQRSwingup(Controller):
    """[summary]

    Parameters
    ----------
    pend : pendsim.sim.Pendulum
        pendulum object
    dt : float
        simulation timestep
    window : int
        the LQR window
    Q : np.ndarray
        State cost array
    R : np.ndarray
        Actuation cost array
    k : float
        swing up gain
    thresh : float, optional
        threshold for swing-up, above this angular value, the controller
        will attempt to swing itself upright; below it, it will perform
        an LQR control strategy.     by default np.pi/4
    """

    def __init__(
        self,
        pend: pendsim.sim.Pendulum,
        dt: float,
        window: int,
        Q: np.ndarray,
        R: np.ndarray,
        k: float,
        thresh: float = np.pi / 4,
    ) -> None:
        self.window = window
        self.A, self.B = self.get_linear_sys(pend.jacA, pend.jacB, dt)
        self.Q = Q
        self.R = R
        self.pend = pend
        self.thresh = thresh
        self.k = k

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        if self.wrapPi(state[2]) > self.thresh:
            action = self.do_swingup(self.pend, state, self.k)
        else:
            action = self.do_lqr(self.window, self.A, self.B, self.Q, self.R, state)

        return action, {}


class NoController(Controller):
    """\"No Controller\" controller. This guy does nothing, 0 action, at
    every timestep.
    """

    def __init__(self):
        pass

    def policy(self, state: np.ndarray, dt: float):
        return float(0), {}


class BangBang(Controller):
    """BangBang control strategy. If the pendulum is within a threshold given
    by `threshold`, push as hard as possible to get it to vertical. When it
    passes over `theta`, push the other way.

    This is what your air conditioner uses and it's probably not ideal for the
    fast dynamics at hand here.

    Parameters
    ----------
    magnitude : float
        magnitude of the push.
    setpoint : float, optional
        setpoint around which to push. 0 by default
    threshold : float, optional
        threshold; if the pendulum falls above the threshold (i.e. if it falls over)
        this prevents the controller from taking continous action. By default np.pi/4
    """

    def __init__(
        self, magnitude: float, setpoint: float = 0, threshold: float = np.pi / 4
    ) -> None:

        self.setpoint = setpoint
        self.magnitude = magnitude
        self.threshold = threshold

    def policy(self, state: np.ndarray, dt: float) -> Tuple[float, dict]:
        error = state[2] - self.setpoint
        action = 0.0
        if error > 0.1 and state[2] < self.threshold:
            action = -self.magnitude
        elif error < -0.1 and state[2] > -self.threshold:
            action = self.magnitude
        else:
            action = 0.0
        return action, {}
