import numpy as np
from matplotlib import patches, text, lines
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


################ VISUALIZATION ####################

class Visualizer(object):
    def __init__(self, 
            data, 
            pend, 
            ls='-', 
            lc='k', 
            frameskip=5, 
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
        self.skip = frameskip
        self.window = window

        ### DISPLAY ###
        self.disp_size = .5
        # Cart Params
        self.cart_w = np.sqrt(self.cart_squish * self.pend.M) * self.disp_size
        self.cart_h = np.sqrt(1/(self.cart_squish) * self.pend.M) * self.disp_size
        # Pendulum Params
        self.p_rad = np.sqrt(self.pend.m) * self.disp_size/3
        # Display Params
        self.xmax = np.stack(self.data['state'].values)[:,0].max() * 1.1
        self.xmin = np.stack(self.data['state'].values)[:,0].min() * 1.1
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

    def display_viz(self):
        '''
        Display (show) the animated visualization. This function calls plt.show()
        '''
        # axis setup
        viz = plt.figure(figsize=self.window)
        ax = plt.axes()
        plt.axis('scaled')
        ax.set_xlim(self.xmin - self.pend.l*2, self.xmax + self.pend.l*2)
        ax.set_ylim(self.ymin, self.ymax)
    
        # matplotlib animate doesn't play nice with dataframes :(
        anim_x = list(np.stack(self.data['state'].values)[:,0])[::self.skip]
        anim_ghost_x = list(np.stack(self.data['setpoint'].values)[:,0])[::self.skip]
        anim_th = list(np.stack(self.data['state'].values)[:,2])[::self.skip]
        anim_ghost_th = list(np.stack(self.data['setpoint'].values)[:,2])[::self.skip]
        anim_f = list(np.stack(self.data['forces'].values)[:])[::self.skip]
        anim_c = list(np.stack(self.data['control action'].values)[:])[::self.skip]
        anim_t = self.data.index.values.tolist()[::self.skip]

        n_frames = len(anim_t)
        # Initialize objects
        cart, mass, line = self._draw_objs(self.lc, self.ls)
        if self.draw_ghost:
            ghostcart, ghostmass, ghostline = self._draw_objs('red', '-')
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


        def init():
            '''
            Function required by matplotlib. Initializes the objects for use by the animator
            '''
            ax.add_patch(cart)
            ax.add_patch(mass)
            ax.add_artist(line)
            ax.add_patch(ext_force)
            ax.add_patch(ctrl_force)
            ax.add_patch(ground)
            ax.add_artist(angle_text)
            ax.add_artist(x_text)
            ax.add_artist(time_text)
            if self.draw_ghost:
                ax.add_patch(ghostcart)
                ax.add_patch(ghostmass)
                ax.add_artist(ghostline)
                return [ground, cart, mass, line, ext_force, 
                        ctrl_force, angle_text, x_text, time_text,
                        ghostcart, ghostmass, ghostline]
            else:
                return [ground, cart, mass, line, ext_force, 
                        ctrl_force, angle_text, x_text, time_text]

        def animate(i):
            '''
            Function required by matplotlib. Runs in a loop during FuncAnimation
            '''
            # draw extern force
            self.draw_force(ext_force, anim_f[i], anim_x[i], 0.6)
            # draw control force
            self.draw_force(ctrl_force, anim_c[i], anim_x[i], 0.3)
            # draw cart
            cartxy_true = (anim_x[i], self.cart_h)
            cartxy_visible = (anim_x[i] - self.cart_w * .5, self.cart_h)
            cart.set_xy(cartxy_visible)
            massxy = (anim_x[i] - self.pend.l * np.sin(anim_th[i]), self.cart_h + self.pend.l * np.cos(anim_th[i]))
            mass.set_center(massxy)
            # draw connecting line
            linexy = np.array([
                [massxy[0], cartxy_true[0]],
                [massxy[1], cartxy_true[1]]
            ])
            line.set_data(linexy)
            # display text
            angle_text.set_text(r"$\theta=$"+str(round(anim_th[i],3)))
            x_text.set_text(r"$x=$" + str(round(anim_x[i],3)))
            time_text.set_text(r"t="+str(round(anim_t[i],3)))

            if self.draw_ghost:
                # cart
                ghostcart.set_xy(
                    (anim_ghost_x[i] - self.cart_w * .5, 
                    self.cart_h)
                )

                # mass
                ghostmass.set_center(
                    (anim_ghost_x[i] - self.pend.l * np.sin(anim_ghost_th[i]), 
                    self.cart_h + self.pend.l * np.cos(anim_ghost_th[i]))
                )
                # line
                ghostline.set_data(np.array([
                    [anim_ghost_x[i] - self.pend.l * np.sin(anim_ghost_th[i]),anim_ghost_x[i]],
                    [self.cart_h + self.pend.l * np.cos(anim_ghost_th[i]),self.cart_h]
                ]))
                return [ground, cart, mass, line, ext_force, 
                        ctrl_force, angle_text, x_text, time_text,
                        ghostcart, ghostline, ghostmass]

            else:
                return [ground, cart, mass, line, ext_force, ctrl_force, angle_text, x_text, time_text]
        
        def run_animation():
            '''
            Function to actually run the animation. Allows pausing on screen
            '''
            anim_running = True
            animation = FuncAnimation(viz, animate, frames=n_frames, init_func=init, blit=True, interval=16)
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