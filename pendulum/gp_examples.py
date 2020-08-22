    '''
    # a sequence of open loop forces
    forces = [
        (50, 1, .5),
        (-100, 6, .25),
        (5, 9, 5),
        (-25, 12, 1),
        (-5, 26, 3),
        (400, 35, .1),
        (-300, 38, .12)]
    # pendulum attributes
    pend_init_state = np.array([0,0,0,0])
    p = Pendulum(
        8,
        2,
        5,
        9.81, 
        cfric=0.1, 
        pfric=0.01, 
        init_state=pend_init_state)

    sim = Simulation(p, 0.001, 50, forces, 0)
    data = sim.simulate(controller.NoController())
    plot = Visualizer(
        data, 
        p, 
        frameskip=25, 
        cart_display_aspect=2,
        save=False, 
        viz_xcenter = -5,
        viz_window_size=(16,9)
    )
    plot.display_viz()
    plot.display_plots()
    plt.show()
    
    # MPC
    forces = [
        (10,1,.5)
    ]
    pend_init_state = np.array([0,0,0,0])
    p = Pendulum(
        8,
        2,
        5,
        9.81,
        init_state = pend_init_state
    )
    sim = Simulation(p, 0.03, 12, forces, 0)
    data = sim.simulate(controller.MPCController(p.init_state, p, 9))
    plot = Visualizer(
        data,
        p,
        frameskip = 1,
        viz_size=10,
        viz_resize_auto=True,
        viz_window_size=(16,9)
    )
    '''
    forces = [
        (10,1,.5)
    ]
    pend_init_state = np.array([0,0,np.pi/4,0])
    p = Pendulum(
        8,
        2,
        5,
        9.81,
        init_state = pend_init_state
    )
    dt = 0.01
    sim = Simulation(p, dt, 15, forces, 0)
    controller = controller.MPCOneShot()
    data = sim.simulate(controller)

    ts = controller.ts
    states = controller.states
    l = len(controller.states)

    CLIP = 25
    BLACKOUT = 300
    PARAM = 0
    WINDOW = 2000

    x = np.atleast_2d(np.linspace(0, ts[-1], num=10000)).transpose()

    y_true = np.array([x[PARAM] for x in states[0:]])
    y = np.array([x[PARAM] for i, x in enumerate(states[0:-BLACKOUT]) if i % CLIP == 0])
    print("y: {}".format(np.shape(y)))

    t_true = np.array([x for x in ts[:]])
    t = np.array([x for i, x in enumerate(ts[0:-BLACKOUT]) if i % CLIP == 0])
    X = np.atleast_2d(t).transpose()
    print("X: {}".format(np.shape(X)))

    # kernel = C(2.0) * ExpSineSquared(length_scale=2.5e4, length_scale_bounds=(1e-4, 1e4), periodicity=0.3, periodicity_bounds=(1e-3, 1e3))
    kernel = C(2.0e-4) * RBF(1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, normalize_y=False)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    # the MSE
    MSE = plt.figure()
    plt.title("x, RBF")
    plt.plot(t_true, y_true, 'k--', label=r'True')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')

    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.3, fc='royalblue', ec='None', label='95% confidence interval')
    plt.xlabel('$t$')
    plt.ylabel('$theta doubledot$')
    plt.legend(loc='upper left')
    plt.show()


    times = controller.t
    # to predict
    x = np.atleast_2d(np.linspace(0, times[-1], num=1000)).transpose()

    # output
    CLIP = 10
    BLACKOUT = 20
    PARAMETER = 2
    WINDOW = 200

    l = len(controller.window)
    print(l)
    y_true = np.array([x[PARAMETER] for x in controller.window[l - WINDOW:]])
    y = np.array([x[PARAMETER] for i, x in enumerate(controller.window[l - WINDOW:(l-BLACKOUT*CLIP)]) if i % CLIP == 0])
    print("y")
    print(np.shape(y))
    # input points
    
    # use fewer inputs for the regression
    t_true = [t for t in controller.times[l - WINDOW:]]
    t = [t for i, t in enumerate(controller.times[l - WINDOW:(l-BLACKOUT*CLIP)]) if i % CLIP == 0]

    X = np.atleast_2d(t).transpose()
    print("X")
    print(np.shape(X))

    ####### KERNEL PARAMS

    # Instantiate a Gaussian Process model
    kernel = C(2.0) * ExpSineSquared(length_scale=2500.0, length_scale_bounds=(1e-2, 5e3), periodicity=0.3, periodicity_bounds=(1e-3, 1e3))
    # kernel = ExpSineSquared(length_scale=10, periodicity=1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, normalize_y=False)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    # the MSE
    MSE = plt.figure()
    plt.plot(t_true, y_true, 'k--', label=r'True')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.3, fc='royalblue', ec='None', label='95% confidence interval')
    plt.xlabel('$t$')
    plt.ylabel('$theta doubledot$')
    plt.legend(loc='upper left')
    plt.show()
    

    plot = Visualizer(
        data,
        p,
        frameskip = 1,
        viz_size=10,
        viz_resize_auto=True,
        viz_window_size=(16,9)
    )   # observations = np.array(controller.window)
    # print(observations)
    # times = controller.times

    
    print(np.shape(observations))
    print(observations[:,2])

    plot.display_plots()
    plt.show()