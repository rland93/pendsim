import numpy as np
from matplotlib import patches, text, lines
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer(object):
    def __init__(self, data, pend, speed=2):
        self.data, self.pend = data, pend
        self.speed = speed
        self.cart_w = np.sqrt(2 * self.pend.M) * .5
        self.cart_h = np.sqrt(1/(2) * self.pend.M) *.5
        # Pendulum Params
        self.p_rad = np.sqrt(self.pend.m) * .5/3
        # Display Params
        self.xmax = self.data.loc(axis=1)['state','x'].values.max() * 1.1
        self.xmin = self.data.loc(axis=1)['state','x'].values.min() * 1.1
        self.ymax = (self.pend.l + self.cart_h) * 1.3
        self.ymin = -self.pend.l * 1.3  

    def _draw_cart(self, cart, mass, line, xi, thetai):
        '''
        Draw a cart using xi, thetai
        '''
        # adjust height when rendering
        cart.set_xy((xi - self.cart_w * .5, self.cart_h))
        # mass xy
        massx = xi - self.pend.l * np.sin(thetai)
        massy = self.cart_h + self.pend.l * np.cos(thetai)
        mass.set_center((massx, massy))
        # line
        linexy = np.array([
            [massx, xi],
            [massy, self.cart_h]
        ])
        line.set_data(linexy)
    
    def _draw_objs(self, lc='black', ls='-'):
        cart = patches.Rectangle(
            (-self.cart_w * 0.5, self.cart_h), 
            width = self.cart_w, 
            height = -self.cart_h, 
            fc = 'white',
            ec = lc,
            ls = ls
        )
        # The pendulum mass
        mass = patches.Circle(
            (0,0), 
            radius=self.p_rad, 
            fc='white', 
            ec= lc,
            ls = ls
        )
        # The line connecting cart to pend mass
        line = lines.Line2D(
            [0,1], [0,1],
            c = 'black',
            linestyle = '-',        
        )
        return cart, mass, line


    def draw_force(self, obj, u, cart_x, ydist):
        if u > 0.0:
            beg = (cart_x - .5 * self.cart_w, ydist * self.cart_h)
            end = (cart_x - .5 * self.cart_w - np.sqrt(.1 * np.abs(u)), ydist * self.cart_h)
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        elif u < 0.0:
            beg = (cart_x + .5 * self.cart_w, ydist * self.cart_h)
            end = (cart_x + .5 * self.cart_w + np.sqrt(.1 * np.abs(u)), ydist * self.cart_h)
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        else:
            obj.set_xy(((0,0), (1,1)))
            obj.set_linewidth(0)
            obj.set_visible(False)

    def animate(self, size=(9,6)):
        # axis setup
        fig, ax = plt.subplots(figsize=size)
        ax.set_xlim(self.xmin - self.pend.l*2, self.xmax + self.pend.l*2)
        ax.set_ylim(self.ymin, self.ymax)
        n_frames = np.floor(len(self.data.index.values.tolist())/self.speed).astype(int) - 10
        # Initialize objects
        cart, mass, line = self._draw_objs()
        # Line for external force
        ext_force = patches.FancyArrow(0,0,1,1, ec='red')
        # Line for control force
        ctrl_force = patches.FancyArrow(0,0,1,1, ec='blue')
        ground = patches.Rectangle((-1000, -2000), 2000, 2000, fc='gray')
        # ground
        ground.set_zorder(-1)
        # Time text
        time_text = text.Annotation('', (4,28), xycoords='axes points')

        def _init():
            '''
            Function required by matplotlib. Initializes the objects for use by the animator
            '''
            ax.add_patch(cart)
            ax.add_patch(mass)
            ax.add_artist(line)
            ax.add_patch(ext_force)
            ax.add_patch(ctrl_force)
            ax.add_patch(ground)
            ax.add_artist(time_text)
            plist = [ground, cart, mass, line, ext_force, ctrl_force, time_text]
            return plist

        def _animate(i):
            retobjs = [ground, cart, mass, line, ext_force, ctrl_force, time_text]
            i = np.floor(i*self.speed).astype(int)
            # draw cart
            state_xi = list(self.data[('state','x')].values)[i]
            state_ti = list(self.data[('state','t')].values)[i]
            self._draw_cart(cart, mass, line, state_xi, state_ti)
            # external force
            self.draw_force(ext_force, list(self.data[('forces','forces')].values)[i], state_xi, 0.6)
            self.draw_force(ctrl_force,list(self.data[('control action','control action')].values)[i],state_xi,0.5)
            time_text.set_text(r"t="+str(round(self.data.index[i],3)))
            return retobjs
        
        return FuncAnimation(fig, _animate, frames=n_frames, init_func=_init, blit=True,interval=30)