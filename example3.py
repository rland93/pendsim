from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib
mpl=False
if mpl:
    matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import cycler

font = {'size'   : 7}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=.5)
matplotlib.rc('')
n = 15
color = plt.cm.terrain(np.linspace(0, 1,n))
matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
fig, ax = plt.subplots(nrows=3, figsize=(4, 3), sharex=True)

# Specify Gaussian Process
kernel = kernels.RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100)
rng = np.random.RandomState(14)
X = rng.uniform(0, 8, 12)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)
ax[0].scatter(X[:, 0], y, c='r', s=25, zorder=10, edgecolors=(0, 0, 0))
ax[0].set_xlim(0, 8)
ax[0].set_ylim(-2, 2)

# Plot prior
X_ = np.linspace(0, 8, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
ax[1].plot(X_, y_mean, 'k', lw=3, zorder=9)
ax[1].fill_between(X_, y_mean - y_std, y_mean + y_std,
                    alpha=0.2, color='k')
y_samples = gp.sample_y(X_[:, np.newaxis], 10)
ax[1].plot(X_, y_samples, lw=1)
ax[1].set_xlim(0, 8)
ax[1].set_ylim(-2, 2)

# Generate data and fit GP
ax[1].scatter(X[:, 0], y, c='r', s=25, zorder=10, edgecolors=(0, 0, 0))
gp.fit(X, y)

# Plot posterior
X_ = np.linspace(0, 8, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
ax[2].plot(X_, y_mean, 'k', lw=3, zorder=9)
ax[2].fill_between(X_, y_mean - y_std, y_mean + y_std,
                    alpha=0.2, color='k')

y_samples = gp.sample_y(X_[:, np.newaxis], 10)
ax[2].plot(X_, y_samples, lw=1)
ax[2].scatter(X[:, 0], y, c='r', s=25, zorder=10, edgecolors=(0, 0, 0))
ax[2].set_xlim(0, 8)
ax[2].set_ylim(-2, 2)
plt.show()
if mpl:
    fig.savefig('./GP.pgf', format='pgf', dpi=200)