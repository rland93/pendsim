from inspect import CO_GENERATOR
import numpy as np
from matplotlib import patches, text, lines
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import math


class Visualizer(object):
    """The visualizer object stores parameters and methods for generating an
    animation of a particular run.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing simulation data
    pend : pendsim.sim.Pendulum
        Pendulum object to visualize. The animation scales with the length and masses
        of the pendulum object. To generate an accurate visualization, just pass the same
        pendulum that was simulated.
    speed : int, optional
        The number of frames to "skip" in the animation. Real-time animation is rather
        difficult with matplotlib, because the draw speed of the animator varies between
        systems. So if your simulation timestep is very small, the animation can run quite
        slowly.      by default 2
    """

    def __init__(self, data: pd.DataFrame, pend, dt, speed: int = 2) -> None:
        self.data, self.pend = data, pend
        self.dt = dt
        self.speed = speed
        self.cart_w = np.sqrt(2 * self.pend.M) * 0.5
        self.cart_h = np.sqrt(1 / (2) * self.pend.M) * 0.5
        # Pendulum Params
        self.p_rad = np.sqrt(self.pend.m) * 0.5 / 3
        # Display Params
        self.xmax = self.data.loc(axis=1)["state", "x"].values.max() * 1.1
        self.xmin = self.data.loc(axis=1)["state", "x"].values.min() * 1.1
        self.ymax = (self.pend.l + self.cart_h) * 1.3
        self.ymin = -self.pend.l * 1.3
        self.f_width = 0.02

    def _draw_cart(
        self,
        cart: patches.Rectangle,
        mass: patches.Circle,
        line: lines.Line2D,
        xi: float,
        thetai: float,
    ):
        """
        Draw cart on `cart`, `mass`, and `line` objects with cart position xi and
        pendulum position thetai.
        """
        # adjust height when rendering
        cart.set_xy((xi - self.cart_w * 0.5, self.cart_h))
        # mass xy
        massx = xi - self.pend.l * np.sin(thetai)
        massy = self.cart_h + self.pend.l * np.cos(thetai)
        mass.set_center((massx, massy))
        # line
        linexy = np.array([[massx, xi], [massy, self.cart_h]])
        line.set_data(linexy)

    def _draw_objs(self, lc="black", ls="-"):
        """Create objects to draw

        Parameters
        ----------
        lc : str, optional
            matplotlib Line color, by default "black"
        ls : str, optional
            matplotlib Line style, by default "-"

        Returns
        -------
        Tuple[patches.Rectangle, patches.Circle, patches.Line2D]
            Tuple of the cart, pendulum, and connecting line objects
        """
        cart = patches.Rectangle(
            (-self.cart_w * 0.5, self.cart_h),
            width=self.cart_w,
            height=-self.cart_h,
            fc="white",
            ec=lc,
            ls=ls,
        )
        # The pendulum mass
        mass = patches.Circle((0, 0), radius=self.p_rad, fc="white", ec=lc, ls=ls)
        # The line connecting cart to pend mass
        line = lines.Line2D(
            [0, 1],
            [0, 1],
            c="black",
            linestyle="-",
        )
        return cart, mass, line

    def draw_force(self, obj, u, cart_x, ydist):
        """Draw the external force on the cart.

        Parameters
        ----------
        obj : matplotlib FancyArrow object
            The force line object
        u : float
            magnitude of force to represent
        cart_x : float
            cart position
        ydist : float
            distance vertically "up" the cart to draw the force. i.e. 0 draws
            the force at the very bottom of the cart along the ground, 1.0 draws
            the force at the very top of the cart
        """
        if u > 0.0:
            beg = (cart_x - 0.5 * self.cart_w, ydist * self.cart_h)
            end = (
                cart_x - 0.5 * self.cart_w - np.sqrt(0.1 * np.abs(u)),
                ydist * self.cart_h,
            )
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        elif u < 0.0:
            beg = (cart_x + 0.5 * self.cart_w, ydist * self.cart_h)
            end = (
                cart_x + 0.5 * self.cart_w + np.sqrt(0.1 * np.abs(u)),
                ydist * self.cart_h,
            )
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(u)))
            obj.set_visible(True)
        else:
            obj.set_xy(((0, 0), (1, 1)))
            obj.set_linewidth(0)
            obj.set_visible(False)

    def animate(
        self,
        pltdata={},
        interval=None,
        draw_fbd=False,
        data_stretch=False,
        figsize=(8, 4.5),
        blit=True,
    ):
        if interval is None:
            interval = self.dt * 1000

        if pltdata:
            fig, ax = plt.subplots(nrows=2, figsize=(figsize[0], figsize[1] * 2))
            ax0, ax1 = ax[0], ax[1]
        else:
            fig, ax0 = plt.subplots(figsize=figsize)

        if data_stretch:
            yls, yus = [], []
            for reskey in pltdata.keys():
                yls.append(self.data[reskey].values.min())
                yus.append(self.data[reskey].values.max())
            yl = min(yls)
            yu = max(yus)
            xl = self.data[reskey].index.min()
            xu = self.data[reskey].index.max()

            ax1.set_ylim((yl, yu))
            ax1.set_xlim((xl, xu))

        # axis setup
        ax0.set_xlim(self.xmin - self.pend.l * 2, self.xmax + self.pend.l * 2)
        ax0.set_ylim(self.ymin, self.ymax)
        ax0.set_aspect("equal")
        n_frames = (
            np.floor(len(self.data.index.values.tolist()) / self.speed).astype(int) - 10
        )
        # Initialize objects
        cart, mass, line = self._draw_objs()
        # Line for external force
        ext_force = patches.FancyArrow(0, 0, 1, 1, ec="red")
        # Line for control force
        ctrl_force = patches.FancyArrow(0, 0, 1, 1, ec="blue")

        if draw_fbd:
            pRx_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            pRy_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            pG_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            cRx_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            cRy_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            cG_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            cN_f = patches.FancyArrow(0, 0, 1, 1, ec="k", zorder=4)
            fbd_draws = (pRx_f, pRy_f, pG_f, cRx_f, cRy_f, cG_f, cN_f)

        ground = patches.Rectangle((-1000, -2000), 2000, 2000, fc="grey")
        # ground
        ground.set_zorder(-1)
        # Time text
        time_text = text.Annotation("", (4, 28), xycoords="axes points")

        plots = []
        for name, attrs in pltdata.items():
            if attrs["type"] == "line":
                (plot,) = ax1.plot(
                    [],
                    [],
                    label=attrs["label"],
                    linestyle=attrs["linestyle"],
                    color=attrs["color"],
                )
                plots.append(plot)
            elif attrs["type"] == "scatter":
                plot = ax1.scatter(
                    [],
                    [],
                    label=attrs["label"],
                    c=attrs["color"],
                    edgecolors=None,
                    marker=".",
                )
                plots.append(plot)
            else:
                raise ValueError("Wrong type or no type given.")

        def _init():
            plist = []
            ax0.add_patch(cart)
            ax0.add_patch(mass)
            ax0.add_artist(line)
            ax0.add_patch(ext_force)
            ax0.add_patch(ctrl_force)
            ax0.add_patch(ground)
            ax0.add_artist(time_text)
            if draw_fbd:
                for fbd in fbd_draws:
                    ax0.add_patch(fbd)
                plist.extend(fbd_draws)
            plist = [ground, cart, mass, line, ext_force, ctrl_force, time_text]
            plist.extend(plots)
            return plist

        def _animate(i):

            i = np.floor(i * self.speed).astype(int)

            retobjs = []
            # limits for y-axis

            if pltdata:
                scyall = [0]
                for (name, attrs), sc in zip(pltdata.items(), plots):
                    l = max(i - attrs["plotpoints"], 0)
                    scx = self.data.index[l:i]
                    scy = self.data[name].values[l:i]
                    if attrs["type"] == "scatter":
                        sc.set_offsets(np.column_stack([scx, scy]))
                    elif attrs["type"] == "line":
                        sc.set_data(scx, scy)
                    scyall.extend(list(scy))
                retobjs.extend(plots)
                yl = min(-0.1, min(scyall))
                yu = max(0.1, max(scyall))
                xl = self.data.index[
                    max(i - max([p["plotpoints"] for p in pltdata.values()]), 0)
                ]
                xu = self.data.index[i] + 1e-5
                if not data_stretch:
                    ax1.set_ylim((yl, yu))
                    ax1.set_xlim((xl, xu))
                ax1.legend(loc=2)

            # draw cart
            state_xi = list(self.data[("state", "x")].values)[i]
            state_ti = list(self.data[("state", "t")].values)[i]
            self._draw_cart(cart, mass, line, state_xi, state_ti)
            # external force
            self.draw_force(
                ext_force,
                list(self.data[("forces", "forces")].values)[i],
                state_xi,
                0.6,
            )
            self.draw_force(
                ctrl_force,
                list(self.data[("control action", "control action")].values)[i],
                state_xi,
                0.5,
            )
            time_text.set_text(r"t=" + str(round(self.data.index[i], 3)))

            # fbds
            if draw_fbd:
                # pend x
                px = state_xi - self.pend.l * np.sin(state_ti)
                # pend y
                py = self.cart_h + self.pend.l * np.cos(state_ti)
                # draw reaction force (pendulum)
                self.draw_pend_fbd(
                    pRx_f,
                    self.data[("forces", "pRx")].values[i],
                    np.array([1, 0]),
                    np.array([px, py]),
                )
                self.draw_pend_fbd(
                    pRy_f,
                    self.data[("forces", "pRy")].values[i],
                    np.array([0, 1]),
                    np.array([px, py]),
                )
                self.draw_pend_fbd(
                    pG_f,
                    self.data[("forces", "pG")].values[i],
                    np.array([0, 1]),
                    np.array([px, py]),
                )
                retobjs.extend((pRx_f, pRy_f, pG_f))
                # draw reaction force (cart)
                cx, cy = state_xi, self.cart_h
                self.draw_cart_fbd(
                    cRx_f,
                    self.data[("forces", "cRx")].values[i],
                    np.array([1, 0]),
                    np.array([cx, cy]),
                )
                self.draw_cart_fbd(
                    cRy_f,
                    self.data[("forces", "cRy")].values[i],
                    np.array([0, 1]),
                    np.array([cx, cy]),
                )
                self.draw_cart_fbd(
                    cG_f,
                    self.data[("forces", "cG")].values[i],
                    np.array([0, 1]),
                    np.array([cx, cy]),
                )
                self.draw_cart_fbd(
                    cN_f,
                    self.data[("forces", "cN")].values[i],
                    np.array([0, 1]),
                    np.array([cx, cy]),
                )

                retobjs.extend((cRx_f, cRy_f, cG_f, cN_f))

            retobjs.extend([ground, cart, mass, line, ext_force, ctrl_force, time_text])
            return retobjs

        anim_running = True

        def onClick(event):
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True

        fig.canvas.mpl_connect("button_press_event", onClick)
        anim = FuncAnimation(
            fig,
            _animate,
            frames=n_frames,
            init_func=_init,
            blit=blit,
            interval=interval,
        )

        return anim

    def draw_cart_fbd(self, obj, f, direc, pos):
        direc = sign(f) * direc
        # initial position plus offset for cart radius
        offsetbeg = pos
        # set beginning/end
        beg, end = offsetbeg, offsetbeg + np.sqrt(self.f_width * np.abs(f)) * direc
        if np.abs(f) > 0:
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(f)))
            obj.set_visible(True)
        else:
            obj.set_visible(False)

    def draw_pend_fbd(self, obj, f, direc, pos):
        direc = sign(f) * direc
        # its initial position plus offset for pend radius
        offsetbeg = pos + direc * self.p_rad
        # set beginning / end
        beg, end = offsetbeg, offsetbeg + np.sqrt(self.f_width * np.abs(f)) * direc
        if np.abs(f) > 0:
            obj.set_xy((beg, end))
            obj.set_linewidth(np.sqrt(np.abs(f)))
            obj.set_visible(True)
        else:
            obj.set_visible(False)


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
