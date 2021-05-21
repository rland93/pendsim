from functools import partialmethod
import numpy as np
from matplotlib import patches, text, lines
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.lib.utils import get_include
import math

class Visualizer(object):
    def __init__(self, 
            data, 
            pend, 
            plotpoints=10,
            ls='-', 
            lc='k', 
            speed=2, 
            draw_ghost=False, 
            cart_squish=2, 
            window=(16,9),
            save=False):
        # sim data
        self.data, self.pend = data, pend
        # styles
        self.ls, self.lc, self.cart_squish = ls, lc, cart_squish
        # draw setpoint ghost
        self.draw_ghost = draw_ghost

        # playback
        self.speed = speed
        self.window = window
        self.plotpoints = plotpoints

        ### DISPLAY ###
        self.disp_size = .5
        # Cart Params
        self.cart_w = np.sqrt(self.cart_squish * self.pend.M) * self.disp_size
        self.cart_h = np.sqrt(1/(self.cart_squish) * self.pend.M) * self.disp_size
        # Pendulum Params
        self.p_rad = np.sqrt(self.pend.m) * self.disp_size/3
        # Display Params
        self.xmax = self.data.loc(axis=1)['state','x'].values.max() * 1.1
        self.xmin = self.data.loc(axis=1)['state','x'].values.min() * 1.1
        self.ymax = (self.pend.l + self.cart_h) * 1.3
        self.ymin = -self.pend.l * 1.3
        # save
        self.save = save

    def _draw_objs(self, lc, ls):
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
            c = self.lc,
            linestyle = self.ls,        
        )
        return cart, mass, line


    def draw_cart(self, cart, mass, line, xi, thetai):
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

    def display_viz(self, pltdata, ghosts=None):
        '''
        Display (show) the animated visualization. This function calls plt.show()
        '''
        # axis setup
        viz, ax = plt.subplots(nrows=2, figsize=self.window)
        ax[0].set_xlim(self.xmin - self.pend.l*2, self.xmax + self.pend.l*2)
        ax[0].set_ylim(self.ymin, self.ymax)
        n_frames = np.floor(len(self.data.index.values.tolist())/self.speed).astype(int) - 10
        # Initialize objects
        cart, mass, line = self._draw_objs(self.lc, self.ls)
        # add ghosts
        ghostobjs = None
        if ghosts is not None:
            ghostobjs = {}
            for gkey, gcolor in ghosts.items():
                ghostobjs[gkey] = self._draw_objs(*gcolor)

        # Line for external force
        ext_force = patches.FancyArrow(0,0,1,1, ec='red')
        # Line for control force
        ctrl_force = patches.FancyArrow(0,0,1,1, ec='blue')
        ground = patches.Rectangle((-1000, -2000), 2000, 2000, fc='lightgrey')
        ground.set_zorder(-1)
        # text
        angle_text = text.Annotation('', (4,4), xycoords='axes points')
        x_text = text.Annotation('',(4,16), xycoords='axes points')
        time_text = text.Annotation('', (4,28), xycoords='axes points')
        scatters = []
        for k, pltattr in pltdata.items():
            sc = ax[1].scatter(
                [],[],
                **pltattr)
            scatters.append(sc)
        def init():
            '''
            Function required by matplotlib. Initializes the objects for use by the animator
            '''
            ax[0].add_patch(cart)
            ax[0].add_patch(mass)
            ax[0].add_artist(line)
            ax[0].add_patch(ext_force)
            ax[0].add_patch(ctrl_force)
            ax[0].add_patch(ground)
            ax[0].add_artist(angle_text)
            ax[0].add_artist(x_text)
            ax[0].add_artist(time_text)
            plist = [ground, cart, mass, line, ext_force, ctrl_force, angle_text, x_text, time_text]
            if ghostobjs is not None:
                for _, gobjs in ghostobjs.items():
                    ax[0].add_patch(gobjs[0])
                    ax[0].add_patch(gobjs[1])
                    ax[0].add_artist(gobjs[2])
                    plist.extend([gobjs[0], gobjs[1], gobjs[2]])
            # add data
            plist.extend(scatters)
            return plist
        
        def animate(i):
            '''
            Runs in a loop during FuncAnimation
            '''
            retobjs = [ground, cart, mass, line, ext_force, ctrl_force, angle_text, x_text, time_text]
            i = np.floor(i*self.speed).astype(int)
            #### Scatter
            if i != 0:
                sci_l = max(i - self.plotpoints, 0)
                sci_r = i
                scx = self.data.index[sci_l:sci_r]
                for d, sc in zip(pltdata, scatters):
                    scy = self.data[d].values[sci_l:sci_r]
                    xy = np.column_stack((scx, scy))
                    sc.set_offsets(xy)

                scxlim = self.data.index[sci_l], self.data.index[sci_r]
                ax[1].set_xlim(scxlim)

                scyall = []
                for d in pltdata:
                    scyall.extend(list(self.data[d].values[sci_l:sci_r]))
                scylimu = max(0.01, max(scyall))
                scyliml = min(0, min(scyall))
                ax[1].set_ylim((scyliml, scylimu))
                ax[1].legend(bbox_to_anchor=(.1, .5), loc='lower right')
            retobjs.extend(scatters)
            # adjust lims


            #### Cart
            

            # if there's a ghost we take its information
            # a ghost key must contain 'x' and 't' subkeys.
            if ghostobjs:
                for gkey, gobjs in ghostobjs.items():
                    g_xi = list(self.data[(gkey, 'x')].values)[i]
                    g_ti = list(self.data[(gkey, 't')].values)[i]
                    self.draw_cart(*gobjs, g_xi, g_ti)
                    retobjs.extend(gobjs)
            # drawcart
            state_xi = list(self.data[('state','x')].values)[i]
            state_ti = list(self.data[('state','t')].values)[i]
            self.draw_cart(cart, mass, line, state_xi, state_ti)

            time_i = self.data.index[i]
            # external force
            self.draw_force(ext_force, list(self.data[('forces','forces')].values)[i], state_xi, 0.6)
            self.draw_force(ctrl_force,list(self.data[('control action','control action')].values)[i],state_xi,0.5)
            # display text
            angle_text.set_text(r"$\theta=$"+str(round(state_ti,3)))
            x_text.set_text(r"$x=$" + str(round(state_xi,3)))
            time_text.set_text(r"t="+str(round(time_i,3)))
            return retobjs

        def run_animation():
            '''
            Function to actually run the animation. Allows pausing on screen
            '''
            anim_running = True
            animation = FuncAnimation(viz, animate, frames=n_frames, init_func=init, blit=False,interval=100)
            def onClick(event):
                nonlocal anim_running
                if anim_running:
                    animation.event_source.stop()
                    anim_running = False
                else:
                    animation.event_source.start()
                    anim_running = True
            viz.canvas.mpl_connect('button_press_event', onClick)
            if self.save:
                animation.save('./video.mp4', fps=30, bitrate=1000)
        # run the animation and show
        run_animation()
        plt.show()
    
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
