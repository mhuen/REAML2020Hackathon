#!/usr/bin/python3
"""
This script collects data from the REST API to obtain an averaged
sensor activation. From these the sensor normalization is computed.
An alternative approach is to use the training data for this.
Ideally, results of both approaches should match up.
"""
import os
import click
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from modules.settings import ALLOWED_SECRET, SERVER, WAYPOINTS, WAY_POINTS
from modules.settings import MODEL_FILE
from modules.controller import RobotController
from modules.localizer import Localizer
from modules.visualization import EventDisplay


@click.command()
@click.option('--n_points', '-n', default=10000,
              help='The number of last points to plot.')
def main(n_points):

    # create the localizer which will asynchronously localize the robot
    localizer = Localizer(
        model_file=MODEL_FILE,
        server=SERVER,
        allowed_secret=ALLOWED_SECRET,
    )

    event_viewer = EventDisplay()
    event_viewer.fig.show()

    t_last = None
    while True:
        t, pred = localizer.localize()
        if t != t_last:
            event_viewer.set_prediction(pred, t)
            t_last = t
        time.sleep(1)


if __name__ == '__main__':
    main()
