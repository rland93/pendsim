# pendsim: an Inverted Pendulum-On-A-Cart Sandbox!

A simple inverse-pendulum simulator, with a module to render animations of the simulation and API for custom controllers. See the [Example Video](https://user-images.githubusercontent.com/33564709/116198178-817dde80-a6ea-11eb-8cdf-e0c53c922416.mp4).

It uses rk45 to simulate a dynamic model of the simple inverse pendulum on a cart: [Inverted Pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum).

For more details on the package features, as well as how it can be used in an educational context, see the pending JOSE submission: ([Markdown](https://github.com/rland93/pendsim/paper.pdf)), ([PDF](https://github.com/rland93/pendsim/paper.pdf))

## Installation
This package is published on PyPi. Install with `pip`:

```bash
pip install pendsim
```

Requires Python 3.

## Documentation

[The Full Package Documentation is available here.](http://rland93.github.io/pendsim/)

## Usage Examples

Several Notebooks are available. They can be run on the cloud using Google Colab:

+ [Advanced Plotting Example](https://colab.research.google.com/github/rland93/pendsim/blob/master/notebooks/tutorial_plot_inline.ipynb)

+ [Linearization Example](https://colab.research.google.com/github/rland93/pendsim/blob/master/notebooks/linearization.ipynb)

+ [PID Tuning Example](https://colab.research.google.com/github/rland93/pendsim/blob/master/notebooks/PID.ipynb)
