#!/usr/bin/python3
"""
This is an example code for TU Dortmund summer-school-2020 Hackathon.

The example shows how the robot can be controlled using the APIs to complete
the whole task.

The competitiors needs only to write the make_prediction function to call their
machine learning trained model. The function should return the position of
the robot as (x,y) cooridnates. The example has a simple path planning
algortihm that can drive the robot around the arena to reach all the waypoints
specified for each team.

All robot frames specified in this code and the whole competition are according
to ROS (robot operating system) standards [for more info check ROS REP 105]
"""

import tensorflow as tf
import requests
import json
import time
import numpy as np
import pandas as pd

from modules.settings import ALLOWED_SECRET, SERVER, WAYPOINTS, WAY_POINTS
from modules.settings import MODEL_FILE, vicon_coords
from modules.controller import RobotController
from modules.localizer import Localizer
from modules.model import ModelWrapper


if __name__ == '__main__':

    # create the localizer which will asynchronously localize the robot
    localizer = Localizer(
        model_file=MODEL_FILE,
        server=SERVER,
        allowed_secret=ALLOWED_SECRET,
    )
    time.sleep(5)

    # create the robot controller
    robot = RobotController(
        localizer=localizer, server=SERVER, allowed_secret=ALLOWED_SECRET)

    # Iterate over waypoints and keep looping until the robot gets a feedback
    # that each waypoint was successfully reached.
    count = 0
    while True:
        way_point = WAY_POINTS[count]
        way_point_coords = vicon_coords[way_point[0]-1, way_point[1]-1]

        # get the robot position
        pos = robot.localize()
        relative_pos = way_point_coords - pos
        print('')
        print('New Step:')
        print('\tway_point', way_point)
        print('\tway_point_coords', way_point_coords)
        print('\tpos', pos)
        print('\trelative_pos', relative_pos)
        print('\tfeedback()', robot.waypoint_feedback())

        # relative_pos = np.array([0, 0])
        # way_point = way_point.flatten()

        # # get the robot position from the latest data and the trained model
        # current_cell = robot.localize().flatten()
        current_cell = robot.convert_to_cell(pos)
        print('\tcurrent_cell', current_cell)

        # abs_distance = np.abs(way_point - current_cell)

        # if current_cell[0] >= way_point[0] and current_cell[1] >= way_point[1] : # north-west quarter
        #     relative_pos[0] = -abs_distance[0]
        #     relative_pos[1] =  abs_distance[1]
        # elif current_cell[0] <= way_point[0] and current_cell[1] >= way_point[1] : # north-east quarter
        #     relative_pos[0] =  abs_distance[0]
        #     relative_pos[1] =  abs_distance[1]
        # elif current_cell[0] <= way_point[0] and current_cell[1] <= way_point[1] : # south-east quarter
        #     relative_pos[0] =  abs_distance[0]
        #     relative_pos[1] = -abs_distance[1]
        # elif current_cell[0] >= way_point[0] and current_cell[1] <= way_point[1] : # south-west quarter
        #     relative_pos[0] = -abs_distance[0]
        #     relative_pos[1] = -abs_distance[1]

        # make sure we actually move:
        norm = np.hypot(relative_pos) / np.sqrt(2)
        if norm < 0.1:
            print('Detected too small step!')
            print('Increasing relative position...')
            relative_pos = relative_pos * (0.2 / norm)

        robot.go_to_relative(relative_pos[0], relative_pos[1])

        waypoint_feedback_count = robot.waypoint_feedback()
        if waypoint_feedback_count > count:
            count = waypoint_feedback_count
            print("Waypoint reached.")
        else:
            print("Robot didn't reach waypoint, retrying.")

        if count == 5:
            break

    print("Task successfully finished, robot reached all waypoints.")
