import numpy as np
import copy
def array_to_kv(level1_key, level2_keys, array):
    data={}
    if array.shape[0] != len(level2_keys):
        raise(ValueError('Level 2 keys are not same length as array: {} vs {}'.format(level2_keys, array.shape[0])))
    for n, name in enumerate(level2_keys):
        key = (level1_key, name)
        val = array[n]
        data[key] = val
    return data

def wrap_pi(state):
    return np.array([state[0],state[1], (state[2] + np.pi) % (2 * np.pi) - np.pi,(state[3] + np.pi) % (2 * np.pi) - np.pi])

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0