#!/usr/bin/python3
"""
This script collects data from the REST API to obtain an averaged
sensor activation. From these the sensor normalization is computed.
An alternative approach is to use the training data for this.
Ideally, results of both approaches should match up.
"""
import os
import click
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from modules.settings import ALLOWED_SECRET, SERVER, WAYPOINTS, WAY_POINTS
from modules.controller import RobotController
from modules.data import get_data_from_data_frame


@click.command()
@click.option('--out_base', '-o',
              default='data/normalization/norm_data_{:08d}.pkl',
              help='The output base name for the data')
@click.option('--save_frequency', '-f', default=1000,
              help='The frequency at which to save.')
@click.option('--size', '-s', default=10000,
              help='The frequency at which to save.')
def main(out_base, save_frequency, size):

    robot = RobotController(
        localizer=None, server=SERVER, allowed_secret=ALLOWED_SECRET)

    # create output directory if it does not exist
    out_dir = os.path.dirname(out_base.format(0))
    if not os.path.exists(out_dir):
        print('Creating output directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    X = np.zeros([size, 23, 15, 10])
    t = np.zeros([size, 1])

    last_time = -float('inf')
    for i in tqdm(range(size)):

        found_new_data = False
        while not found_new_data:

            # get current data frame
            try:
                df_data = robot.get_current_data_frame()
            except Exception as e:
                print(e)
                continue

            # convert to table-structure and extract average time
            X_i, t_i = get_data_from_data_frame(df_data)

            # check if data is new
            if np.abs(t_i - last_time) > 0.05:
                last_time = t_i
                found_new_data = True

                X[i] = X_i
                t[i] = t_i

            # sleep for a little to not over do it with GET requests
            time.sleep(1.)

        # save results away
        if i != 0 and (i % save_frequency == 0 or i == size - 1):
            pickle_data = [t[:i], X[:i]]
            data_pickle_file = out_base.format(i)
            with open(data_pickle_file, 'wb') as handle:
                pickle.dump(pickle_data, handle, protocol=2)


if __name__ == '__main__':
    main()
