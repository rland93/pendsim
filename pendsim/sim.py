import numpy as np
from scipy import integrate
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
from typing import Tuple
from pendsim.utils import array_to_kv


##### Utility Functions
class Pendulum(object):
    """Class for a pendulum object

    Parameters
    ----------
    M : float
        Cart weight, in kg
    m : float
        Pendulum weight, in kg
    l : float
        Length of pendulum, in m
    g : float, optional
        Gravitational constant, in m/s^2, by default 9.81
    cfric : float, optional
        Cart viscous friction coefficient, by default 0.1
    pfric : float, optional
        Pendulum viscous friction coefficient, by default 0.05
    initial_state : np.ndarray, optional
        Initial pendulum state, by default np.array([0, 0, 0, 0])

    Attributes
    ----------
    jacA : np.array
        The jacobian of the state transition matrix A

    jacB : np.array
        The jacobian of the control matrix B
    """

    def __init__(
        self,
        M: float,
        m: float,
        l: float,
        g: float = 9.81,
        cfric: float = 0.1,
        pfric: float = 0.05,
        initial_state: np.ndarray = np.array([0, 0, 0, 0]),
    ) -> None:

        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.cfric = cfric
        self.pfric = pfric
        self.y_0 = initial_state
        # jacobians of the system
        # linearized about the upward position x=[0,0,0,0]
        self.jacA = np.array(
            [
                [0, 1.0, 0, 0],
                [0, 0, -self.m * self.g / self.M, 0.0],
                [0, 0, 0, 1.0],
                [0, 0, self.g * (self.m * self.M) / (self.l * self.M), 0.0],
            ]
        )
        self.jacB = np.array([[0.0], [1.0 / self.M], [0.0], [-1.0 / (self.l * self.M)]])

    def system_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Solve the system dynamics in the form xdot = A\\x

        Because this function is called by `solve_ivp`, we package the control
        input `u` with the state; i.e. the state vector is [x, xdot, theta, thetadot, u]

        input `t` is unused, because the pendulum is modeled as an LTI system.

        Parameters
        ----------
        t : float
            time
        state : np.ndarray
            state, including cart force `u`.

        Returns
        -------
        np.ndarray
            x dot
        """
        # unpack variables
        xd, theta, td, u = state[1], state[2], state[3], state[4]
        m, M, l, g, cfric, pfric = (
            self.m,
            self.M,
            self.l,
            self.g,
            self.cfric,
            self.pfric,
        )
        cost, sint = np.cos(theta), np.sin(theta)
        # solve for derivatives
        xdd = (g * m * sint * cost + u - m * l * td * td * sint) / (
            M + m - m * cost * cost
        )
        tdd = (g * sint + xdd * cost) / l
        # frictions
        xdd += -cfric * xd
        tdd += -pfric * td
        return np.array([xd, xdd, td, tdd, u])

    def solve(
        self, dt: float, system_state: np.ndarray, u: float, solve_args: dict = {}
    ) -> Tuple[np.ndarray, float]:
        """Given a system state, solve the ivp over interval `d

        Parameters
        ----------
        dt : float
            Timestep over which to solve ivp
        system_state : np.ndarray
            the 4-tuple system state [x, xdot, theta, thetadot]
        u : float
            external force applied to base
        solve_args : dict, optional
            arguments supplied to the solver, by default {}

        Returns
        -------
        Tuple[np.ndarray, float]
            tuple of (4-tuple system state, external force)
        """
        # roll state and u into a single array
        y = np.empty((5))
        y[:4] = system_state
        y[4] = u
        # solve system
        sol = integrate.solve_ivp(self.system_dynamics, (0, dt), y, **solve_args)
        return sol.y[:4, -1], sol.y[4, -1]

    def get_energy(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Get energy of a state.

        Parameters
        ----------
        state : np.ndarray
            system state

        Returns
        -------
        Tuple[float, float, float]
            tuple of (kinetic, potential, total) energies.
        """
        xd, t, td = state[1], state[2], state[3]
        ke = (
            0.5 * (self.m + self.M) * xd * xd
            - self.m * self.l * xd * td * np.cos(t)
            + 0.5 * self.m * self.l * self.l * td * td
        )
        # potential energy
        pe = self.m * self.g * self.l * np.cos(t)
        return ke, pe, ke + pe

    def calculate_reaction_forces(
        self, state: np.ndarray, xdd: float, tdd: float
    ) -> tuple:
        """Get reaction forces on each body (cart and pendulum). This can be
        used to construct or visualize the forces that are applied to each body

        Parameters
        ----------
        state : np.ndarray
            system state
        xdd : float
            cart acceleration
        tdd : float
            angular acceleration

        Returns
        -------
        tuple
            (pend reaction force x-dir, pend reaction force y-dir, pend G force,
            cart reaction force x-dir, cart reaction force y-dir, cart G force,
            cart normal force)
        """
        x, xd, theta, td = state[0], state[1], state[2], state[3]
        sint, cost = np.sin(theta), np.cos(theta)
        pRx = self.m * (xdd + self.l * td * td * sint - self.l * tdd * cost)
        pRy = self.m * (-self.l * td * td * cost - self.l * tdd * sint)
        pG = self.m * self.g
        cRy = -pRy
        cRx = -pRx
        cG = self.M * self.g
        cN = -cG - cRy
        return pRx, pRy, pG, cRy, cRx, cG, cN


class Simulation(object):
    """Simulation object

    Parameters
    ----------
    dt : float
        simulation timestep in seconds
    t_final : float
        final simulation time in seconds (simulation happens over the interval [0, t_final])
    force : callable
        a time-dependent forcing function. Must be in the form `f(t) = u` where t is the
        supplied time and u is the corresponding force applied to the cart.
    noise_scale : np.ndarray, optional
        scale of gaussian noise to add to the system state. Is a 4-tuple where each value corresponds
        to the variance of the noise added to each state value, by default None
    """

    def __init__(
        self, dt: float, t_final: float, force: callable, noise_scale: np.ndarray = None
    ):
        self.dt = dt  # time step
        self.t_final = t_final  # end at or before this time
        self.force = force  # forcing function
        self.noise_scale = noise_scale

    def simulate(self, pendulum: Pendulum, controller) -> pd.DataFrame:
        """Simulate the system with the parameters stored in the `Simulation` object.

        User supplies a pendulum and a controller, so that the same simulation conditions
        (timestep, noise, external force, etc) can be tested against multiple pendulums and
        controllers.

        Parameters
        ----------
        pendulum : Pendulum
            Pendulum object under simulation
        controller : [type]
            Controller to run over the simulation

        Returns
        -------
        pd.DataFrame
            The results of the simulation. Calling e.g. results.columns will show all fields
            recorded during the simulation. This is a 2-level Multi-Index dataframe.
        """
        # initial state
        state = pendulum.y_0
        # initialize data
        datas = defaultdict(list)
        statelabels = ["x", "xd", "t", "td"]
        # time
        times = np.arange(start=0, stop=self.t_final, step=self.dt)

        for k, t in tqdm(enumerate(times), total=len(times)):
            data = {}
            force = self.force(t)

            # Control policy
            data.update(array_to_kv("state", statelabels, state))
            if self.noise_scale is not None:
                noisy_state = state + np.random.normal(0, scale=self.noise_scale)
                data.update(array_to_kv("measured state", statelabels, noisy_state))
                data.update(
                    array_to_kv("noise variance", statelabels, self.noise_scale)
                )
                action, controller_data = controller.policy(noisy_state, self.dt)
            else:
                action, controller_data = controller.policy(state, self.dt)
            data.update(controller_data)

            # Simulation Data
            (
                data[("energy", "kinetic")],
                data[("energy", "potential")],
                data["energy", "total"],
            ) = pendulum.get_energy(state)
            data[("forces", "forces")] = force
            data[("control action", "control action")] = action

            # Add action to force
            force += action
            # Calculate Reaction Forces
            deriv_input = np.empty((5,))
            deriv_input[
                0:4,
            ] = state
            deriv_input[
                0,
            ] = action
            deriv = pendulum.system_dynamics(0, deriv_input)
            xdd, tdd = deriv[2], deriv[4]
            pRx, pRy, pG, cRy, cRx, cG, cN = pendulum.calculate_reaction_forces(
                state, xdd, tdd
            )
            data[("state", "xdd")] = xdd
            data[("state", "tdd")] = tdd
            data[("forces", "pRx")] = pRx
            data[("forces", "pRy")] = pRy
            data[("forces", "pG")] = pG
            data[("forces", "cRy")] = cRy
            data[("forces", "cRx")] = cRx
            data[("forces", "cG")] = cG
            data[("forces", "cN")] = cN

            state, _ = pendulum.solve(self.dt, state, force)
            for k, v in data.items():
                datas[k].append(v)

        return pd.DataFrame(datas, index=times)

    def simulate_multiple(self, pendulums, controllers, parallel=True):
        """Simulate pendulums and controllers in a list. This method sees significant speedup
        in parallelization.

        Parameters
        ----------
        pendulums : list
            List of pendulums to simulate
        controllers : list
            List of controllers to simulate
        parallel : bool, optional
            Whether to simulate in parallel across multiple cores, by default True

        Returns
        -------
        pd.DataFrame
            Dataframe of results. This dataframe basically stacks each run in a multi-index,
            where the level0 indices are the multiple runs of the simulation and the level1 indices
            are the timesteps of each run.

        Raises
        ------
        ValueError
            pendulums and controllers must be 1:1, i.e. each pendulum must be supplied with a
            controller, even if controllers or pendulums used across runs are the same.
        """
        if len(pendulums) != len(controllers):
            raise ValueError(
                "pendulums and controllers must have same length. len(pendulums)={}, len(controllers)={}".format(
                    len(pendulums), len(controllers)
                )
            )
        pc = list(zip(pendulums, controllers))
        if parallel:
            results = Pool().map(self.runsim, pc)
            return pd.concat(results, axis=0, keys=list(range(len(results))))
        else:
            allresults = []
            for pendulum, controller in pc:
                results = self.simulate(pendulum, controller)
                allresults.append(results)
            return pd.concat(allresults, axis=0, keys=list(range(len(results))))

    def runsim(self, pend_cont_tuple):
        return self.simulate(*pend_cont_tuple)
