import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from math import sin, cos, pi
from scipy.integrate import solve_ivp

def force_pendulum(t, state, forces, params, forces_output, energy, momentum):
    '''
    state: [x, th, xdot, thdot]
    returns: [xdot, thdot, xddot, thddot]
    '''
    ### unroll these for readability

    # constants
    M = params[0]
    m = params[1]
    l = params[2]
    g = params[3]

    # state vars at current time k
    x_k = state[0]
    t_k = state[1]
    xd_k = state[2]
    td_k  = state[3]
    sin_t = np.sin(t_k)
    cos_t = np.cos(t_k)
    F = 0.0

    ### check for forces: read through the list of force tuples and apply each
    if len(forces) > 0:
        for force in forces:
            # apply forces & record
            if (t > force[1] and not t > force[2]):
                F = force[0]
                force_output.append((t, F))
            else: 
                F = 0.0
                force_output.append((t, F))

    ### compute energy E = T + U
    # velocity of cart^2
    v_c2 = xd_k * xd_k 
    # velocity of pendulum^2
    v_p2 = v_c2 - 2 * l * xd_k * td_k * cos_t + l*l*td_k*td_k 
    # energy
    E = 0.5 * M * v_c2 + .5 * m * v_p2 + m*g*l*cos_t
    # record energy
    energy.append((t, E))

    ### computer linear momentum
    p_cart = M * xd_k
    p_pend = m * (xd_k + td_k * sin_t * l)
    momentum.append((t, p_cart, p_pend))

    
    ### solve xxdot, thddot (componentwise)
    xdd = (F + m*g*sin_t *cos_t - m * l * td_k * td_k * sin_t) / (M+m-m*cos_t*cos_t)
    tdd = xdd * cos_t / l + g * sin_t / l
    

    '''
    ### solve xddot, thddot
    A = np.array([
        [M + m  , -m * l * cos_t],
        [-cos_t    , l          ]
    ])
    b = np.array([
        [F - m * l * td_k * td_k],
        [g * sin_t]
    ])
    solution = np.linalg.solve(A,b)
    xdd = solution[0][0]
    tdd = solution[1][0]
    '''

    ### return d/dt state
    return [xd_k, td_k, -xdd, tdd]

######## CONSTANTS ###############
CART_Y = 1 # height of the cart
CART_DISPLAY_WIDTH = 2 # width of the cart (display)

S_TFINAL = 10 # Simulation time
S_TPS = 120 # Simulation ticks per second

########## PARAMETERS #############

# x, theta, xdot, thetadot
init_state = [0, 0, 0, 0]
# forces: list of tuples, (magnitude, start t, end t)
forces = [(10, 1, 2)]
# M, m, l, g
params = [8, 5, 6, 9.81]
'''
# cart damping, pendulum damping
damping = []
'''
# record forces are written here
force_output = []
energy = []
momentum = []
########### Solve IVP #############
solution = solve_ivp(
    force_pendulum, 
    [0,S_TFINAL], 
    init_state, 
    t_eval=np.linspace(0,S_TFINAL,num=(S_TPS*S_TFINAL)), 
    args=(forces, params, force_output, energy, momentum),
    method='BDF'
    )
print(solution)


########## ANIMATE #############


# Set up figure

gs = GridSpec(2,5)

fig = plt.figure(figsize=(20,8))


# Animation window
ax1 = fig.add_subplot(gs[:,:2])
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
cart = patches.Rectangle((-CART_DISPLAY_WIDTH*.5,CART_Y), width=CART_DISPLAY_WIDTH, height=-CART_Y, ec='black', fc='salmon')
mass = patches.Circle((0,0), radius=np.sqrt(params[1])*.14, fc='skyblue', ec='black')
line = patches.FancyArrow(0,0,1,1)
time_text = text.Annotation('', (4,4), xycoords='axes points')
ax1.grid(True)

# Initialize animation object
def init():
    ax1.add_patch(mass)
    ax1.add_patch(cart)
    ax1.add_artist(time_text)
    ax1.add_patch(line)
    return [mass, cart, time_text, line]

# Animate ith frame
def animate(i):
    x = solution.y[0][i]
    th = solution.y[1][i]
    massxy = (x + params[2] * sin(th), CART_Y + params[2] * cos(th))
    cartxy_visible = (x-CART_DISPLAY_WIDTH*.5, CART_Y)
    cartxy_true = (x, CART_Y)
    mass.set_center(massxy)
    cart.set_xy(cartxy_visible)
    line.set_xy((massxy, cartxy_true))
    time_text.set_text("t="+str(solution.t[i]))
    return [mass, cart, time_text, line]

# Plot: x, theta, xdot, thetadot
ax2 = fig.add_subplot(gs[0,2:])
ax2.set_xlabel("t")
ax2.plot(solution.t, solution.y[0], label="x")
ax2.plot(solution.t, solution.y[1], label="theta")
ax2.plot(solution.t, solution.y[2], label="x dot")
ax2.plot(solution.t, solution.y[3], label="theta dot")
ax2.legend()

# Plot: forces, energy, momentum
ax3 = fig.add_subplot(gs[1,2:])
force_t = [f[0] for f in force_output]
force_forces = [f[1] for f in force_output]
ax3.scatter(force_t, force_forces, label='forces')
energy_t = [e[0] for e in energy]
energy_energies = [e[1] for e in energy]
# ax3.plot(energy_t, energy_energies, label='energy')
momentum_t = [m[0] for m in momentum]
momentum_cart = [m[1] for m in momentum]
momentum_pend = [m[2] for m in momentum]
momentum_total = [(m[1] + m[2]) for m in momentum]
# ax3.plot(momentum_t, momentum_cart, label='cart momentum')
# ax3.plot(momentum_t, momentum_pend, label='pendulum momentum')
# ax3.plot(momentum_t, momentum_total, label='total momentum')
ax3.legend()

animation = FuncAnimation(fig, animate, len(solution.y[0]), init_func=init, blit=True, interval=(1000/S_TPS))
# animation.save('video.mp4', fps=30)
plt.show()