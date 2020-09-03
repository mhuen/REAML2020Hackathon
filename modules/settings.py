"""
Global settings for the TU Dortmund summer-school-2020 Hackathon are saved here
"""
import numpy as np


w1 = [18, 5]
w2 = [18, 9]
w3 = [14, 7]
w4 = [10, 5]
w5 = [9, 9]
w6 = [8, 7]
w7 = [5, 4]
w8 = [4, 6]

###############################################################################
###############################################################################
SERVER = "http://s17.phynetlab.com/"

# This is a secret key that will be given to each team to get access to
# control the robot during the second task.
ALLOWED_SECRET = '5ebe2294acd0e0f08eab7690d2a6ee69'

# Specify the waypoint provided to each team (5 points)
WAYPOINTS = [w1, w6, w4, w7, w5]  # Team 1 waypoints
# WAYPOINTS = [w1, w6, w5, w7, w4]  # Team 2 waypoints
# WAYPOINTS = [w2, w6, w4, w7, w8]  # Team 3 waypoints
# WAYPOINTS = [w8, w3, w4, w7, w5]  # Team 4 waypoints
# WAYPOINTS = [w8, w3, w5, w7, w6]  # Team 5 waypoints
WAY_POINTS = np.array(WAYPOINTS)

MODEL_FILE = 'data/models/model_test_0000'
###############################################################################
###############################################################################


vicon_coords = np.array([
       [[-11.185,   7.575],
        [-11.185,   6.575],
        [-11.185,   5.575],
        [-11.185,   4.575],
        [-11.185,   3.575],
        [-11.185,   2.575],
        [-11.185,   1.575],
        [-11.185,   0.575],
        [-11.185,  -0.425],
        [-11.185,  -1.425],
        [-11.185,  -2.425],
        [-11.185,  -3.425],
        [-11.185,  -4.425],
        [-11.185,  -5.425],
        [-11.185,  -6.425]],

       [[-10.185,   7.575],
        [-10.185,   6.575],
        [-10.185,   5.575],
        [-10.185,   4.575],
        [-10.185,   3.575],
        [-10.185,   2.575],
        [-10.185,   1.575],
        [-10.185,   0.575],
        [-10.185,  -0.425],
        [-10.185,  -1.425],
        [-10.185,  -2.425],
        [-10.185,  -3.425],
        [-10.185,  -4.425],
        [-10.185,  -5.425],
        [-10.185,  -6.425]],

       [[ -9.185,   7.575],
        [ -9.185,   6.575],
        [ -9.185,   5.575],
        [ -9.185,   4.575],
        [ -9.185,   3.575],
        [ -9.185,   2.575],
        [ -9.185,   1.575],
        [ -9.185,   0.575],
        [ -9.185,  -0.425],
        [ -9.185,  -1.425],
        [ -9.185,  -2.425],
        [ -9.185,  -3.425],
        [ -9.185,  -4.425],
        [ -9.185,  -5.425],
        [ -9.185,  -6.425]],

       [[ -8.185,   7.575],
        [ -8.185,   6.575],
        [ -8.185,   5.575],
        [ -8.185,   4.575],
        [ -8.185,   3.575],
        [ -8.185,   2.575],
        [ -8.185,   1.575],
        [ -8.185,   0.575],
        [ -8.185,  -0.425],
        [ -8.185,  -1.425],
        [ -8.185,  -2.425],
        [ -8.185,  -3.425],
        [ -8.185,  -4.425],
        [ -8.185,  -5.425],
        [ -8.185,  -6.425]],

       [[ -7.185,   7.575],
        [ -7.185,   6.575],
        [ -7.185,   5.575],
        [ -7.185,   4.575],
        [ -7.185,   3.575],
        [ -7.185,   2.575],
        [ -7.185,   1.575],
        [ -7.185,   0.575],
        [ -7.185,  -0.425],
        [ -7.185,  -1.425],
        [ -7.185,  -2.425],
        [ -7.185,  -3.425],
        [ -7.185,  -4.425],
        [ -7.185,  -5.425],
        [ -7.185,  -6.425]],

       [[ -6.185,   7.575],
        [ -6.185,   6.575],
        [ -6.185,   5.575],
        [ -6.185,   4.575],
        [ -6.185,   3.575],
        [ -6.185,   2.575],
        [ -6.185,   1.575],
        [ -6.185,   0.575],
        [ -6.185,  -0.425],
        [ -6.185,  -1.425],
        [ -6.185,  -2.425],
        [ -6.185,  -3.425],
        [ -6.185,  -4.425],
        [ -6.185,  -5.425],
        [ -6.185,  -6.425]],

       [[ -5.185,   7.575],
        [ -5.185,   6.575],
        [ -5.185,   5.575],
        [ -5.185,   4.575],
        [ -5.185,   3.575],
        [ -5.185,   2.575],
        [ -5.185,   1.575],
        [ -5.185,   0.575],
        [ -5.185,  -0.425],
        [ -5.185,  -1.425],
        [ -5.185,  -2.425],
        [ -5.185,  -3.425],
        [ -5.185,  -4.425],
        [ -5.185,  -5.425],
        [ -5.185,  -6.425]],

       [[ -4.185,   7.575],
        [ -4.185,   6.575],
        [ -4.185,   5.575],
        [ -4.185,   4.575],
        [ -4.185,   3.575],
        [ -4.185,   2.575],
        [ -4.185,   1.575],
        [ -4.185,   0.575],
        [ -4.185,  -0.425],
        [ -4.185,  -1.425],
        [ -4.185,  -2.425],
        [ -4.185,  -3.425],
        [ -4.185,  -4.425],
        [ -4.185,  -5.425],
        [ -4.185,  -6.425]],

       [[ -3.185,   7.575],
        [ -3.185,   6.575],
        [ -3.185,   5.575],
        [ -3.185,   4.575],
        [ -3.185,   3.575],
        [ -3.185,   2.575],
        [ -3.185,   1.575],
        [ -3.185,   0.575],
        [ -3.185,  -0.425],
        [ -3.185,  -1.425],
        [ -3.185,  -2.425],
        [ -3.185,  -3.425],
        [ -3.185,  -4.425],
        [ -3.185,  -5.425],
        [ -3.185,  -6.425]],

       [[ -2.185,   7.575],
        [ -2.185,   6.575],
        [ -2.185,   5.575],
        [ -2.185,   4.575],
        [ -2.185,   3.575],
        [ -2.185,   2.575],
        [ -2.185,   1.575],
        [ -2.185,   0.575],
        [ -2.185,  -0.425],
        [ -2.185,  -1.425],
        [ -2.185,  -2.425],
        [ -2.185,  -3.425],
        [ -2.185,  -4.425],
        [ -2.185,  -5.425],
        [ -2.185,  -6.425]],

       [[ -1.185,   7.575],
        [ -1.185,   6.575],
        [ -1.185,   5.575],
        [ -1.185,   4.575],
        [ -1.185,   3.575],
        [ -1.185,   2.575],
        [ -1.185,   1.575],
        [ -1.185,   0.575],
        [ -1.185,  -0.425],
        [ -1.185,  -1.425],
        [ -1.185,  -2.425],
        [ -1.185,  -3.425],
        [ -1.185,  -4.425],
        [ -1.185,  -5.425],
        [ -1.185,  -6.425]],

       [[ -0.185,   7.575],
        [ -0.185,   6.575],
        [ -0.185,   5.575],
        [ -0.185,   4.575],
        [ -0.185,   3.575],
        [ -0.185,   2.575],
        [ -0.185,   1.575],
        [ -0.185,   0.575],
        [ -0.185,  -0.425],
        [ -0.185,  -1.425],
        [ -0.185,  -2.425],
        [ -0.185,  -3.425],
        [ -0.185,  -4.425],
        [ -0.185,  -5.425],
        [ -0.185,  -6.425]],

       [[  0.815,   7.575],
        [  0.815,   6.575],
        [  0.815,   5.575],
        [  0.815,   4.575],
        [  0.815,   3.575],
        [  0.815,   2.575],
        [  0.815,   1.575],
        [  0.815,   0.575],
        [  0.815,  -0.425],
        [  0.815,  -1.425],
        [  0.815,  -2.425],
        [  0.815,  -3.425],
        [  0.815,  -4.425],
        [  0.815,  -5.425],
        [  0.815,  -6.425]],

       [[  1.815,   7.575],
        [  1.815,   6.575],
        [  1.815,   5.575],
        [  1.815,   4.575],
        [  1.815,   3.575],
        [  1.815,   2.575],
        [  1.815,   1.575],
        [  1.815,   0.575],
        [  1.815,  -0.425],
        [  1.815,  -1.425],
        [  1.815,  -2.425],
        [  1.815,  -3.425],
        [  1.815,  -4.425],
        [  1.815,  -5.425],
        [  1.815,  -6.425]],

       [[  2.815,   7.575],
        [  2.815,   6.575],
        [  2.815,   5.575],
        [  2.815,   4.575],
        [  2.815,   3.575],
        [  2.815,   2.575],
        [  2.815,   1.575],
        [  2.815,   0.575],
        [  2.815,  -0.425],
        [  2.815,  -1.425],
        [  2.815,  -2.425],
        [  2.815,  -3.425],
        [  2.815,  -4.425],
        [  2.815,  -5.425],
        [  2.815,  -6.425]],

       [[  3.815,   7.575],
        [  3.815,   6.575],
        [  3.815,   5.575],
        [  3.815,   4.575],
        [  3.815,   3.575],
        [  3.815,   2.575],
        [  3.815,   1.575],
        [  3.815,   0.575],
        [  3.815,  -0.425],
        [  3.815,  -1.425],
        [  3.815,  -2.425],
        [  3.815,  -3.425],
        [  3.815,  -4.425],
        [  3.815,  -5.425],
        [  3.815,  -6.425]],

       [[  4.815,   7.575],
        [  4.815,   6.575],
        [  4.815,   5.575],
        [  4.815,   4.575],
        [  4.815,   3.575],
        [  4.815,   2.575],
        [  4.815,   1.575],
        [  4.815,   0.575],
        [  4.815,  -0.425],
        [  4.815,  -1.425],
        [  4.815,  -2.425],
        [  4.815,  -3.425],
        [  4.815,  -4.425],
        [  4.815,  -5.425],
        [  4.815,  -6.425]],

       [[  5.815,   7.575],
        [  5.815,   6.575],
        [  5.815,   5.575],
        [  5.815,   4.575],
        [  5.815,   3.575],
        [  5.815,   2.575],
        [  5.815,   1.575],
        [  5.815,   0.575],
        [  5.815,  -0.425],
        [  5.815,  -1.425],
        [  5.815,  -2.425],
        [  5.815,  -3.425],
        [  5.815,  -4.425],
        [  5.815,  -5.425],
        [  5.815,  -6.425]],

       [[  6.815,   7.575],
        [  6.815,   6.575],
        [  6.815,   5.575],
        [  6.815,   4.575],
        [  6.815,   3.575],
        [  6.815,   2.575],
        [  6.815,   1.575],
        [  6.815,   0.575],
        [  6.815,  -0.425],
        [  6.815,  -1.425],
        [  6.815,  -2.425],
        [  6.815,  -3.425],
        [  6.815,  -4.425],
        [  6.815,  -5.425],
        [  6.815,  -6.425]],

       [[  7.815,   7.575],
        [  7.815,   6.575],
        [  7.815,   5.575],
        [  7.815,   4.575],
        [  7.815,   3.575],
        [  7.815,   2.575],
        [  7.815,   1.575],
        [  7.815,   0.575],
        [  7.815,  -0.425],
        [  7.815,  -1.425],
        [  7.815,  -2.425],
        [  7.815,  -3.425],
        [  7.815,  -4.425],
        [  7.815,  -5.425],
        [  7.815,  -6.425]],

       [[  8.815,   7.575],
        [  8.815,   6.575],
        [  8.815,   5.575],
        [  8.815,   4.575],
        [  8.815,   3.575],
        [  8.815,   2.575],
        [  8.815,   1.575],
        [  8.815,   0.575],
        [  8.815,  -0.425],
        [  8.815,  -1.425],
        [  8.815,  -2.425],
        [  8.815,  -3.425],
        [  8.815,  -4.425],
        [  8.815,  -5.425],
        [  8.815,  -6.425]],

       [[  9.815,   7.575],
        [  9.815,   6.575],
        [  9.815,   5.575],
        [  9.815,   4.575],
        [  9.815,   3.575],
        [  9.815,   2.575],
        [  9.815,   1.575],
        [  9.815,   0.575],
        [  9.815,  -0.425],
        [  9.815,  -1.425],
        [  9.815,  -2.425],
        [  9.815,  -3.425],
        [  9.815,  -4.425],
        [  9.815,  -5.425],
        [  9.815,  -6.425]],

       [[ 10.815,   7.575],
        [ 10.815,   6.575],
        [ 10.815,   5.575],
        [ 10.815,   4.575],
        [ 10.815,   3.575],
        [ 10.815,   2.575],
        [ 10.815,   1.575],
        [ 10.815,   0.575],
        [ 10.815,  -0.425],
        [ 10.815,  -1.425],
        [ 10.815,  -2.425],
        [ 10.815,  -3.425],
        [ 10.815,  -4.425],
        [ 10.815,  -5.425],
        [ 10.815,  -6.425]]
    ]
)