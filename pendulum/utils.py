import numpy as np

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
    unwrap = lambda v: np.arctan2(np.sin(v), np.cos(v))
    return np.array([
        state[0],
        state[1], 
        unwrap(state[2]),
        unwrap(state[3])
        ])

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0