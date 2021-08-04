from matplotlib import get_backend
import pendsim, pytest
import numpy as np


@pytest.fixture
def pend():
    return pendsim.sim.Pendulum(3.0, 1.0, 2.0)


@pytest.fixture
def BangBang():
    return pendsim.controller.BangBang(0, 10)


@pytest.fixture
def PID():
    return pendsim.controller.PID(1, 1, 1)


@pytest.fixture
def LQR(pend):
    Q = np.diag([1, 1, 1, 1])
    R = np.atleast_2d([1])
    return pendsim.controller.LQR(pend, 0.01, 10, Q, R)


@pytest.fixture
def NoController():
    return pendsim.controller.NoController()


@pytest.fixture
def policy_args():
    return (np.array([0.1, 0.0, 2.11, 0.02]), 0.02)


def test_policy_BangBang(BangBang, policy_args):
    policy_output = BangBang.policy(*policy_args)
    assert type(policy_output) == tuple
    assert type(policy_output[0]) == float or type(policy_output[0]) == np.float64
    assert type(policy_output[1]) == dict


def test_policy_PID(PID, policy_args):
    policy_output = PID.policy(*policy_args)
    assert type(policy_output) == tuple
    assert type(policy_output[0]) == float or type(policy_output[0]) == np.float64
    assert type(policy_output[1]) == dict


def test_policy_LQR(LQR, policy_args):
    policy_output = LQR.policy(*policy_args)
    assert type(policy_output) == tuple
    assert type(policy_output[0]) == float or type(policy_output[0]) == np.float64
    assert type(policy_output[1]) == dict


def test_policy_NoController(NoController, policy_args):
    policy_output = NoController.policy(*policy_args)
    assert type(policy_output) == tuple
    assert type(policy_output[0]) == float or type(policy_output[0]) == np.float64
    assert type(policy_output[1]) == dict
