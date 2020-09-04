"""
Robot Controller class based on the example code provided
for the TU Dortmund summer-school-2020 Hackathon.
"""
from __future__ import print_function, division

import sys
from datetime import datetime, timezone
import requests
import json
import time
import numpy as np
import pandas as pd


class RobotController:
    def __init__(
            self,
            localizer,
            server='http://s17.phynetlab.com/',
            allowed_secret='5ebe2294ecd0e0f08eab7690d2a6ee69',
            ):
        """Instanciate the Robot Controller.

        Parameters
        ----------
        localizer : Localizer instance
            A robot localizer instance. This instance is used to localize
            the robaot asynchronously.
        server : str, optional
            Description
        allowed_secret : str, optional
            Description
        """
        self.localizer = localizer
        self.url_drive = server + allowed_secret + '/robot/action'
        self.url_waypoints = server + allowed_secret + '/robot/waypoints'
        self.url_data = server + allowed_secret + '/current_values'

    def go_forward(self, x):
        """
        This function will drive the robot forward relative to its current
        position and orientation.
        """
        payload = {'action': 'forward', 'value': str(x)}
        return requests.post(self.url_drive, json=payload)

    def turn(self, theta):
        """
        This function will turn the robot around its center.
        Signs of turn directions are according to ROS standards.
        Z-axis points up, positive turn rotation are counter-clockwise and
        negative turn direction is clockwise.

        theta should be specified in angles.
        """
        payload = {'action': 'turn', 'value': str(theta)}
        return requests.post(self.url_drive, json=payload)

    def go_to_relative(self, x, y):
        """
        This function will drive the robot to an (x,y) position relative to the
        current robot position but with the orientation of the origin frame.
        So, the robot will move to an (x,y) position relative to its position
        but as if it was oriented as the origin frame. Also the robot will
        orient itself as the origin frame when it reaches the goal point.

        This is function is used witin this example to drive the robot between
        cells (hence waypoints).
        """
        payload = {'action': 'go_to_relative', 'x': str(x), 'y': str(y)}
        return requests.post(self.url_drive, json=payload)

    def waypoint_feedback(self):
        """
        This functions returns a feedback about the number of waypoints reached
        by the robot.
        """
        msg = requests.get(self.url_waypoints)
        return msg.json()['waypoints_reached']

    def localize(self):
        """Localize robot.

        Returns
        -------
        array_like
            The predicted position in meters.
            Shape: [2]
        """

        # time offset used in data class
        t_offset = 2459067

        # get current time
        current_time = pd.Timestamp(
            datetime.now(timezone.utc)).to_julian_date() - t_offset

        # convert to seconds
        current_time *= 24 * 60 * 60

        # Floor data is delayed, so make sure to catch up
        t, pos = self.localizer.localize()
        print('time:', t - current_time)
        print('Localizing: waiting to catch up time delay...', end='')
        sys.stdout.flush()
        while not t >= current_time:
            t, pos = self.localizer.localize()
            time.sleep(0.5)
        print('Caught up!')

        return pos

    def get_current_data_frame(self):
        """Get Current sensor values as a pandas dataframe
        """

        # read the current values from the APIs
        response = requests.request("GET", self.url_data)
        df_frame = pd.read_json(response.text.encode('utf8'))
        return df_frame

    def convert_to_cell(self, predict_vicon):
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

    def path_planner(self, goal):
        """
        This function will drive the robot to a relative position in (x,y)
        coordinates.

        No need to explicitly write a path planner. Calling "go_to_relative"
        API will use the robot Planner to drive the robot to the desired point.

        Other APIs ("forward" and "turn") can give more control over the robot
        bevahiour. It's up to the competitors to decide it they want to use
        these APIs and how to use it as the "go_to_relative" API should be
        enough to finish the whole taks in an easy way.
        """
        self.go_to_relative(goal[0], goal[1])
