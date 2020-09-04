'''
Plotting functions using matplotlib
'''
from datetime import datetime, timezone
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .settings import vicon_coords, WAY_POINTS


def convert_to_cell(predict_vicon):
        '''
        This function takes a measurement in origin frame (x,y), convert it
        to a cell number and return it as (strip_id, node_id)
        '''
        def translate(value, leftMin, leftMax, rightMin, rightMax):
            # Figure out how 'wide' each range is
            leftSpan = leftMax - leftMin
            rightSpan = rightMax - rightMin

            # Convert the left range into a 0-1 range (float)
            valueScaled = float(value - leftMin) / float(leftSpan)

            # Convert the 0-1 range into a value in the right range.
            return rightMin + (valueScaled * rightSpan)

        predict_cell = np.array([0, 0])
        predict_cell[0] = predict_vicon[0] + 11.185
        predict_cell[1] = translate(predict_vicon[1], -15+7.575, 7.575, 15, 0)
        predict_cell = np.round(predict_cell)

        return predict_cell + 1


class EventDisplay:
    ''' Matplotlib Event Display'''

    def __init__(self, ax=None, n_points=10, **kwargs):
        '''
        Create an event display for a detector.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            if None, the current axes will be used, if there is no current
            axes, a new one will be created.
        n_points : int, optional
            Number of previous points to draw.
        **kwargs
            Description
        **kwargs are passed to ax.pcolormesh
        '''
        self.ax = ax or plt.gca()
        self.fig = self.ax.figure
        self.fig.set_size_inches(16, 10, forward=True)
        nan_point = np.array([[0., 0.]])
        self.points = [nan_point for i in range(n_points)]

        vicon_coords_reshape = np.reshape(vicon_coords, [23*15, 2])

        self.title = self.ax.set_title(
            'Time: {:3.3f} | Position {:3.3f} {:3.3f} | Cell {} {}'.format(
                0., 0., 0., 0., 0.))
        self.coords = self.ax.scatter(
            vicon_coords_reshape[:, 0], vicon_coords_reshape[:, 1],
            marker='o', color='0.8', label='Sensors',
        )
        self.scatters = []
        for i, point in enumerate(self.points):
            alpha = (float(i) / n_points)**2
            if i > 1:
                color = '0.2'
            else:
                color = 'green'
            scat = self.ax.scatter(
                point[:, 0], point[:, 1], marker='+', s=500,
                color=color, linewidth=5, alpha=alpha,
            )
            self.scatters.append(scat)

        # draw waypoints
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(WAY_POINTS)):
            way_point = WAY_POINTS[i]
            coords = vicon_coords[way_point[0] - 1, way_point[1] - 1] - 0.5
            rect = matplotlib.patches.Rectangle(
                coords, 1., 1., color=colors[i], alpha=0.3, zorder=-2,
                label='Waypoint {}'.format(i+1),
            )
            self.ax.add_patch(rect)

        self.ax.set_aspect(1)
        self.ax.legend()
        self.ax.set_xlabel('Vicon X [meter]')
        self.ax.set_ylabel('Vicon Y [meter]')

    def update(self):
        """ redraw the display now """
        self.ax.figure.canvas.draw()

    def set_prediction(self, y_pred, t):
        if len(y_pred.shape) == 1:
            y_pred = y_pred[None, ...]

        cell = convert_to_cell(y_pred[0])

        # time offset used in data class
        t_offset = 2459067

        # get current time
        current_time = pd.Timestamp(
            datetime.now(timezone.utc)).to_julian_date() - t_offset

        # convert to seconds
        current_time *= 24 * 60 * 60

        # update points
        self.points.pop(0)
        self.points.append(y_pred)
        for scat, point in zip(self.scatters, self.points):
            scat.set_offsets(point)
            scat.changed()
        self.title.set_text(
            'Time: {:3.3f}s | Position {:3.3f}m {:3.3f}m | Cell {} {}'.format(
                t - current_time, y_pred[0, 0], y_pred[0, 1],
                cell[0], cell[1]))
        self.update()
