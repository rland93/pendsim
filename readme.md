# Inverse Pendulum Simulation

A simple inverse-pendulum simulator. ![(Example Video)](https://user-images.githubusercontent.com/33564709/116198178-817dde80-a6ea-11eb-8cdf-e0c53c922416.mp4)


[Package Documentation](http://rland93.github.io/pendulum/)

It includes implementations for controllers:
+ PID
+ Bang Bang
+ LQR

As well as options for custom implementations.

Example Notebooks:

+ [No Controller (Simulation Only)](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/nocontroller.ipynb)
+ [PID Controller](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/PID.ipynb)
+ [LQR (With Gaussian Process Prediction)](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/lqr_gpr.ipynb)

It uses rk45 to simulate a dynamic model of the simple inverse pendulum on a cart: [Inverted Pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum).
