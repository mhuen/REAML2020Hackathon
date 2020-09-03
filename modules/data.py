"""
Utility functions for data manipulation and handling
"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from .settings import vicon_coords


KEYS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']


def append_vicon_coords(X):
    """Append coordinates to a data tensor

    Parameters
    ----------
    X : array_like
        The data tensor to which the coordinates will be appended.
        Shape: [n_events, 23, 15, n_features]

    Returns
    -------
    array_like
        The new data tensor with coordinates added in the last dimension
        Shape: [n_events, 23, 15, n_features + 2]
    """
    coords = np.tile(vicon_coords[None, ...], [len(X), 1, 1, 1])
    X = np.concatenate((X, coords), axis=-1)
    return X


def get_data_from_data_frame(df_data):
    """Get data (X, t) from a data frame.

    Parameters
    ----------
    df : pd.DataFrame
        The json read data frame from the sensor data.
        This can for instance be collected by calling `get_current_data_frame`
        of a RobotController instance.

    Returns
    -------
    np.ndarray
        The data tensor. Sensors without a readout are set to zeros.
        Shape: [23, 15, 10]
    float
        The average (offset) time of the sensor readout timestamps.
    """
    X = np.zeros([23, 15, 10])
    for i, key in enumerate(KEYS):
        X[df_data.strip_id - 1, df_data.node_id - 1, i] = df_data[key]

    # Calculate average frame time
    time_stamps = df_data.timestamp
    time_i = []
    offset = 2459067
    for time_stamp in time_stamps:
        time_i.append(time_stamp.to_julian_date() - offset)
    t = np.mean(time_i) * 24 * 60 * 60

    return X, t


def read_data(file):
    """Read data from a training or test hdf5 file.

    Parameters
    ----------
    file : str
        The path to the file.

    Returns
    -------
    list of dict
        A list of frames. A frame is a dictionary with the keys:
            'data': pd.DataFrame
                The sensor data as a pd.DataFrame
            'vicon_x' array_like
                The x-coordinate of the true position. (if available)
            'vicon_y' array_like
                The y-coordinate of the true position. (if available)
    np.ndarray
        The true robot position at each frame.
        Note: if the data is not available, this will just contain zeros.
        Shape: [num_frames, 2]
    np.ndarray
        The average time of each frame.
        Shape: [num_frames, 1]
    np.ndarray
        The sensor readout data. Sensors without a readout are set to zeros.
        Shape: [num_frames, 23, 15, 10]
    """

    # Read a single file as pandas DataFrame
    df = pd.read_csv(file)

    frames = []

    y = np.zeros([len(df), 2])
    t = np.zeros([len(df), 1])
    X = np.zeros([len(df), 23, 15, 10])

    # Generate a single frame for each row
    for index, row in tqdm(df.T.items(), total=len(df)):

        if 'vicon_x' in row:
            y[index] = (row.vicon_x, row.vicon_y)

        # Create dictionary with constant values (i.e. labels and frame number)
        frame = {col: row[col] for col in df.columns if col != 'data'}

        # Create pandas.DataFrame from Json String located in the 'data' column
        df_i = pd.read_json(row['data'])
        frame['data']  = df_i

        X_i, t_i = get_data_from_data_frame(df_i)
        X[index] = X_i
        t[index] = t_i

        frames.append(frame)

    return frames, y, t, X


def get_vel_and_acc(pos, t):
    """Compute Velocity and Acceleration

    Parameters
    ----------
    pos : array_like
        The array of positions.
        Shape: [n_pos, 2]
    t : array_like
        The array of time stamps.
        Shape: [n_pos, 1]

    Returns
    -------
    array_like
        The magnitude of the velocity vector.
        Shape: [n_pos, 1]
    array_like
        The magnitude of the acceleration vector.
        Shape: [n_pos, 1]
    array_like
        The signed magnitude of the parrallel acceleration component.
        Shape: [n_pos, 1]
    array_like
        The signed magnitude of the perpendicular acceleration component.
        Shape: [n_pos, 1]
    """
    # compute delta ts
    dt = np.diff(t, axis=0)

    # compute velocities
    vel = np.zeros_like(pos)
    vel[1:] = np.diff(pos, axis=0) / dt
    vel_abs = np.hypot(vel[:, 0], vel[:, 1])[:, None] / np.sqrt(2.)

    # compute accelerations
    acc = np.zeros_like(pos)
    acc[1:] = np.diff(vel, axis=0) / dt
    acc_abs = np.hypot(acc[:, 0], acc[:, 1])[:, None] / np.sqrt(2.)

    # compute parallel and perpendicular contributions
    eps = 1e-6
    vel_norm = vel_abs + eps
    acc_parallel = np.sum(
        vel * acc, axis=1, keepdims=True) / vel_norm
    acc_perp = ((vel[:, 1]*acc[:, 0]
                 - vel[:, 0]*acc[:, 1])[:, None]
               ) / vel_norm

    return vel_abs, acc_abs, acc_parallel, acc_perp