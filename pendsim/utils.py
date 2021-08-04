from typing import Hashable
import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable

def array_to_kv(level1_key: Hashable, level2_keys: Iterable[Hashable], array: np.ndarray) -> dict:
    data={}
    if array.shape[0] != len(level2_keys):
        raise(ValueError('Level 2 keys are not same length as array: {} vs {}'.format(level2_keys, array.shape[0])))
    for n, name in enumerate(level2_keys):
        key = (level1_key, name)
        val = array[n]
        data[key] = val
    return data

def sign(x: float) -> float:
    if x >= 0: return 1.0
    else: return -1.0

def impulse_func(a:float=0.5, t_offset:float=4.0) -> callable:
    return lambda t: 1/(a*np.sqrt(np.pi)) * np.exp(-((t-t_offset)/a)**2)

def sine_func(period=1) -> callable:
    return lambda t: np.sin(period*t)

def plot_input_func(ifunc:callable, t_final:float, ax: plt.Axes) -> None:
    x = np.linspace(0, t_final, 1000)
    y = ifunc(x)
    ax.plot(x, y)