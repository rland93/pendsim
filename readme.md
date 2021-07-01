# Inverse Pendulum Simulation

A simple inverse-pendulum simulator, with a module to render animations of the simulation and API for custom controllers. ![(Example Video)](https://user-images.githubusercontent.com/33564709/116198178-817dde80-a6ea-11eb-8cdf-e0c53c922416.mp4)

[Package Documentation](http://rland93.github.io/pendulum/)

Example Notebooks (Links to Google Colab):

+ [No Controller (Simulation Only)](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/nocontroller.ipynb)
+ [PID Controller](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/PID.ipynb)
+ [PID From Scratch](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/PD_workshop.ipynb)
+ [LQR](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/LQR.ipynb)
+ [LQR with Unscented Kalman Filter for State Estimation](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/lqr_state_estimation.ipynb)
+ [LQR (With Gaussian Process Prediction)](https://colab.research.google.com/github/rland93/pendulum/blob/master/notebooks/lqr_gpr.ipynb)



It uses rk45 to simulate a dynamic model of the simple inverse pendulum on a cart: [Inverted Pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum).
