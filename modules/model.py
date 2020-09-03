"""This file defines the model class.
"""
import os
import tensorflow as tf
import numpy as np
import pickle
from scipy.optimize import minimize
from tqdm import tqdm_notebook as tqdm

from egenerator.utils import basis_functions

from .data import (
    get_vel_and_acc,
    get_data_from_data_frame,
    append_vicon_coords,
)


# -------------------------------------------
# Define Hard-coded Regularization Parameters
# -------------------------------------------
reg_hard_acc_abs = 3
reg_soft_acc_abs = 1.0
reg_params_acc_abs = np.array([
    # max, hard, soft
    [0.555, reg_hard_acc_abs, reg_soft_acc_abs],  # 1
    [0.488, reg_hard_acc_abs, reg_soft_acc_abs],  # 2
    [0.453, reg_hard_acc_abs, reg_soft_acc_abs],  # 3
    [0.421, reg_hard_acc_abs, reg_soft_acc_abs],  # 4
    [0.383, reg_hard_acc_abs, reg_soft_acc_abs],  # 5
    [0.356, reg_hard_acc_abs, reg_soft_acc_abs],  # 6
    [0.318, reg_hard_acc_abs, reg_soft_acc_abs],  # 7
    [0.296, reg_hard_acc_abs, reg_soft_acc_abs],  # 8
    [0.270, reg_hard_acc_abs, reg_soft_acc_abs],  # 9
    [0.247, reg_hard_acc_abs, reg_soft_acc_abs],  # 10
    [0.235, reg_hard_acc_abs, reg_soft_acc_abs],  # 11
    [0.231, reg_hard_acc_abs, reg_soft_acc_abs],  # 12
    [0.225, reg_hard_acc_abs, reg_soft_acc_abs],  # 13
    # [0.28, reg_hard_acc_abs, reg_soft_acc_abs],  # 14
    # [0.26, reg_hard_acc_abs, reg_soft_acc_abs],  # 15
    # [0.25, reg_hard_acc_abs, reg_soft_acc_abs],  # 16
    # [0.24, reg_hard_acc_abs, reg_soft_acc_abs],  # 17
    # [0.23, reg_hard_acc_abs, reg_soft_acc_abs],  # 18
    # [0.22, reg_hard_acc_abs, reg_soft_acc_abs],  # 19
    # [0.21, reg_hard_acc_abs, reg_soft_acc_abs],  # 20
    # [0.20, reg_hard_acc_abs, reg_soft_acc_abs],  # 21
    # [0.19, reg_hard_acc_abs, reg_soft_acc_abs],  # 22
    # [0.18, reg_hard_acc_abs, reg_soft_acc_abs],  # 23
    # [0.175, reg_hard_acc_abs, reg_soft_acc_abs],  # 24
    # [0.17, reg_hard_acc_abs, reg_soft_acc_abs],  # 25
])
reg_hard_acc_perp_abs = 2
reg_soft_acc_perp_abs = 0.5
reg_params_acc_perp_abs = np.array([
    # max, hard, soft
    [0.745, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 1
    [0.659, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 2
    [0.564, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 3
    [0.500, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 4
    [0.472, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 5
    [0.436, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 6
    [0.416, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 7
    [0.407, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 8
    [0.405, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 9
    [0.390, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 10
    [0.373, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 11
    [0.372, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 12
    [0.372, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 13
    # [0.28, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 14
    # [0.26, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 15
    # [0.25, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 16
    # [0.24, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 17
    # [0.23, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 18
    # [0.22, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 19
    # [0.21, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 20
    # [0.20, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 21
    # [0.19, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 22
    # [0.18, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 23
    # [0.175, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 24
    # [0.17, reg_hard_acc_perp_abs, reg_soft_acc_perp_abs],  # 25
])
reg_params_acc_parallel_abs = np.array([
    # max neg, max pos, hard, soft
    [0.856, 1.072, 2., 0.5],  # 1
    [0.764, 1.022, 2., 0.5],  # 2
    [0.718, 0.938, 2., 0.5],  # 3
    [0.684, 0.867, 2., 0.5],  # 4
    [0.652, 0.794, 2., 0.5],  # 5
    [0.611, 0.729, 2., 0.5],  # 6
    [0.553, 0.657, 2., 0.5],  # 7
    [0.503, 0.601, 2., 0.5],  # 8
    [0.451, 0.550, 2., 0.5],  # 9
    [0.416, 0.500, 2., 0.5],  # 10
    [0.379, 0.480, 2., 0.5],  # 11
    [0.351, 0.462, 2., 0.5],  # 12
    [0.332, 0.454, 2., 0.5],  # 13
    # [0.23, 0.34, 2., 0.5],  # 14
    # [0.22, 0.33, 2., 0.5],  # 15
    # [0.20, 0.325, 2., 0.5],  # 16
    # [0.19, 0.32, 2., 0.5],  # 17
    # [0.17, 0.31, 2., 0.5],  # 18
    # [0.17, 0.30, 2., 0.5],  # 19
    # [0.17, 0.30, 2., 0.5],  # 20
    # [0.15, 0.29, 2., 0.5],  # 21
    # [0.15, 0.275, 2., 0.5],  # 22
    # [0.15, 0.275, 2., 0.5],  # 23
    # [0.145, 0.27, 2., 0.5],  # 24
    # [0.14, 0.26, 2., 0.5],  # 25
])
# -------------------------------------------


class ModelWrapper:

    """Model Wrapper Class

    The model wrapper class is a wrapper around a keras model. Additional
    utilities include:

        - Data normalization
            Normalization data is loaded from disk

        - Postprocessing
            Postprocessing step for a regularized likelihood fit
    """

    def __init__(
            self, model,
            sensor_norm_file=None,
            kde_file='data/kdes/kde_vel_abs.pkl',
            spline_file='data/splines/spline_vel_abs.pkl',
            ):
        """Initialize the ModelWrapper class.

        Parameters
        ----------
        model : keras Model or str
            The trained & loaded keras model or path to the saved model.
        sensor_norm_file : str, optional
            The path to the normalization data that will be used to normalize
            the sensor data. If it is not provided, then it is assumed, that
            model is given as a path to the model directory and that in that
            directory a file named normalization.txt exists from which the
            sensor normalization model file is read out.
        kde_file : str, optional
            The path to the KDE file which is used for the regularziation
            terms.
        spline_file : str, optional
            The path to the spline file to use ofr the regularization.

        Raises
        ------
        ValueError
            Description
        """

        if sensor_norm_file is None:
            if isinstance(model, str):
                sensor_meta = os.path.join(model, 'normalization.txt')
                with open(sensor_meta, "r") as text_file:
                    lines = text_file.readlines()
                assert len(lines) == 1, lines
                sensor_norm_file = lines[0].replace('\n', '')
            else:
                raise ValueError('If not providing the model as a directory, '
                                 'you must provide a sensor norm file path')
        # load model if needed
        if isinstance(model, str):
            meta_file = os.path.join(model, 'meta.pkl')
            model = tf.keras.models.load_model(
                model, custom_objects={'loss_fun': loss_fun})
        else:
            meta_file = None

        self.model = model
        self.sensor_norm_file = sensor_norm_file
        self.kde_file = kde_file
        self.spline_file = spline_file

        # load normalization data
        with open(self.sensor_norm_file, 'rb') as handle:
            sensor_bias, sensor_std = pickle.load(handle, encoding='latin1')

        # load meta data
        if meta_file is not None and os.path.exists(meta_file):
            with open(meta_file, 'rb') as handle:
                self.config = pickle.load(handle, encoding='latin1')
        else:
            self.config = {
                'add_coordinates': False,
            }
        print('Loaded Model with settings:')
        for key, value in self.config.items():
            print('\t{}: {}'.format(key, value))

        # load KDEs
        with open(self.kde_file, 'rb') as handle:
            self.kdes = pickle.load(handle, encoding='latin1')

        # load Splines
        with open(self.spline_file, 'rb') as handle:
            self.splines = pickle.load(handle, encoding='latin1')

        self.sensor_bias = sensor_bias
        self.sensor_std = sensor_std

    def check_data(self, X, mean_tol=2., std_tol=15.):
        """Check distribution of sensor data and see it fits to norm data.

        Parameters
        ----------
        X : array_like
            The unnormalized data tensor.
        mean_tol : float, optional
            The tolerance of shift in mean measured in sigmas.
        std_tol : float, optional
            The tolerance for the standard deviation. This is a factor
            multiplied/divided to the expected normalized standard devation
            of 1. Anything below 1./std_tol or above 1*std_tol will be flagged.

        Returns
        -------
        bool
            Boolean indicating whether everything seems ok (True) or if there
            might be an issue (false). Note that this is not a definitive test,
            it is just a first check and precautionary step.
        """

        # make a copy
        X = np.array(X)
        X_norm = self.normalize(X)

        # remove NaNs
        X[~np.isfinite(X)] = 0
        X_norm[~np.isfinite(X_norm)] = 0

        mean = np.nanmean(X_norm, axis=0)
        abs_mean = np.abs(mean)
        std = np.nanstd(X_norm, axis=0)

        # check if mean is reasonable
        passed_test = True
        if (abs_mean > mean_tol).any():
            passed_test = False
            print('Mean failed:')
            print(abs_mean[abs_mean > mean_tol])

        if ((std > 1*std_tol).any() or (std < 1. / std_tol).any()):
            passed_test = False

            mask = np.logical_or(std > 1*std_tol, std < 1./std_tol)
            mask = np.logical_and(mask, std > 0.)
            print('Std failed:')
            print(std[mask])

        if not passed_test:
            print('===================')
            print('=== Test Failed ===')
            print('===================')
            print('Mean: [should be around zero]')
            print(np.mean(mean, axis=(0, 1)))
            print()
            print('Std: [should be around 1]')
            print(np.std(std, axis=(0, 1)))
            print('===================')
        return passed_test

    def format_data(self, X):
        """Format input data to necessary format.

        Parameters
        ----------
        X : array_like
            The unnormalized data tensor of sensor readout.
            Shape: [n_pos, 23, 15, 10]

        Returns
        -------
        array_like
            The correctly formated and normalized data as used by the
            NN model.
        """

        # normalize the data
        X_norm = self.normalize(X)

        # remove NaNs
        X_norm[~np.isfinite(X_norm)] = 0

        # limit sensor data
        X_norm = X_norm[..., 6:]

        # add coordinates if necessary
        if self.config['add_coordinates']:
            X_norm = append_vicon_coords(X_norm)

        return X_norm

    def normalize(self, X):
        """Normalize a data tensor

        Parameters
        ----------
        X : array_like
            The unnormalized data tensor.

        Returns
        -------
        array_like
            The normalized data tensor.
        """
        X_norm = np.array(X)
        X_norm = (X_norm - self.sensor_bias) / self.sensor_std
        return X_norm

    def post_process(self, mu, sigma, r, t, x0=None, use_kde=False,
                     pre_fitting_size=None, verbose=True):
        """Postprocess estimated positions.

        Run regularized MLE to estimate regularized positions. This can utilize
        the connection between points and the fact that the roboter should move
        continously.

        Parameters
        ----------
        mu : array_like
            The estimated fit positions by the neural network.
            Shape: [n_pos, 2]
        sigma : array_like
            The estimate uncertainty of the asymmetric Gaussian.
            Shape: [n_pos, 1]
        r : array_like
            The estimate asymmetry parameter of the asymmetric Gaussian.
            Shape: [n_pos, 1]
        t : array_like
            The time stamps of each position.
            Shape: [n_pos, 1]
        x0 : array_like, optional
            An initial guess for the best fit positions. If none is provided,
            the provided `mu` will be used.
            Shape: [n_pos, 2]
        use_kde : bool, optional
            Wheter or not to use a kde for the regularization.
            --> Not implemented yet!
        pre_fitting_size : int, optional
            If provided, a smaller chunk of the given size will be optimized
            first. These are then combined and then everything is optimized
            at once.
        verbose : bool, optional
            Print out result object.

        Returns
        -------
        array_like
            The updated and post-processed position estimates.
            Shape: [n_pos, 2]
        """
        return postprocess(
            mu=mu, sigma=sigma, r=r, t=t, x0=x0,
            use_kde=use_kde, kdes=None, splines=None, #self.splines,
            pre_fitting_size=pre_fitting_size, verbose=verbose,
        )

    def get_posterior(self, X, check_data=True):
        """Get the posterior prediction from the neural network.

        Parameters
        ----------
        X : np.ndarray
            The unnormalized data tensor of sensor readout.
            Shape: [n_pos, 23, 15, 10]
        check_data : bool, optional
            If true, the data will be checked to see if there are any
            signs for mis-calibration.

        Returns
        -------
        np.ndarray
            The predicted positions.
            Shape: [n_pos, 2]
        np.ndarray
            The predicted sigma of the asymmetric Gaussian.
            Shape: [n_pos, 2]
        np.ndarray
            The predicted r of the asymmetric Gaussian.
            Shape: [n_pos, 2]
        """

        # check if data seems reasonable
        if check_data and not self.check_data(X):
            print('[WARNING]: Check if data format/calibration is correct!')

        # normalize and properly format data
        X_norm = self.format_data(X)

        # run model prediction
        y_pred = self.model.predict(X_norm)

        mu = y_pred[..., :2]
        sigma = y_pred[..., 2:4]
        r = y_pred[..., 4:]

        return mu, sigma, r

    def predict_from_data_frame(self, df):
        """Predict the roboter positions from a DataFrame.

        Note: this does not run the post-processing.

        Parameters
        ----------
        df : pd.DataFrame
            The json read data frame from the sensor data.
            This can for instance be collected by calling
            `get_current_data_frame` of a RobotController instance.

        Returns
        -------
        array_like
            The predicted roboter positions.
        """
        # extract data from the frame
        X, t = get_data_from_data_frame(df)

        return self.predict(X, t, run_post_processing=False)

    def predict(self, X, t, run_post_processing=True, check_data=True,
                **kwargs):
        """Predict the roboter positions from data tensors.

        Parameters
        ----------
        X : np.ndarray
            The unnormalized data tensor of sensor readout.
            Shape: [n_pos, 23, 15, 10]
        t : array_like
            The time stamps of each position.
            Shape: [n_pos, 1]
        run_post_processing : bool, optional
            If True, the post_processing step will be run.
        check_data : bool, optional
            If true, the data will be checked to see if there are any
            signs for mis-calibration.
        **kwargs
            Keyword arguments that are passed on to the post-processing step.

        Returns
        -------
        array_like
            The predicted roboter positions.
        """

        # get estimated posterior from the NN model
        mu, sigma, r = self.get_posterior(X, check_data=check_data)

        if run_post_processing:
            mu = self.post_process(mu, sigma, r, t, **kwargs)

        return mu

    def predict_iteratively(self, X, t, run_post_processing=False, step_size=1,
                            check_data=True, **kwargs):
        """Predict the roboter positions from data tensors.

        This is done in an iterative fashion: `step_size` new frames are taken
        initially and then reconstructed and post-processed. Then an additional
        `step_size` amount of frames are added up until everything is included.

        Parameters
        ----------
        X : np.ndarray
            The unnormalized data tensor of sensor readout.
            Shape: [n_pos, 23, 15, 10]
        t : array_like
            The time stamps of each position.
            Shape: [n_pos, 1]
        run_post_processing : bool, optional
            If True, the post_processing step will be run.
        step_size : int, optional
            Description
        check_data : bool, optional
            If true, the data will be checked to see if there are any
            signs for mis-calibration.
        **kwargs
            Keyword arguments that are passed on to the post-processing step.

        Returns
        -------
        array_like
            The predicted roboter positions.
        """

        # get estimated posterior from the NN model
        mu, sigma, r = self.get_posterior(X, check_data=check_data)

        mu_best = np.array(mu)

        stop_indices = np.arange(2, len(t) + step_size, step=step_size)
        for stop_index in tqdm(stop_indices):
            mu_split = mu[:stop_index]
            sigma_split = sigma[:stop_index]
            r_split = r[:stop_index]
            t_split = t[:stop_index]
            x0_split = mu_best[:stop_index]

            # run iterative post-processing
            mu_best_split = self.post_process(
                mu_split, sigma_split, r_split, t_split,
                x0=x0_split, verbose=False, **kwargs)

            # update mu_best
            mu_best[:stop_index] = mu_best_split

        return mu_best


def loss_fun(y_true, y_pred):
    """Keras loss function via Asymmetric Gaussian.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    tf.tensor
        The scalar loss value.
    """
    mu = y_pred[..., :2]
    sigma = y_pred[..., 2:4]
    r = y_pred[..., 4:6]
    loss = tf.reduce_mean(
        -basis_functions.tf_log_asymmetric_gauss(
            y_true, mu=mu, sigma=sigma, r=r)
    )
    #oss = tf.reduce_mean(
    #   -basis_functions.tf_log_gauss(
    #       y_true, mu=mu, sigma=sigma)
    #
    #loss = tf.reduce_sum((y_true - y_pred)**2)
    return loss


def likelihood(mu_fit, mu, sigma, r, t, use_kde=False,
               kdes=None, splines=None):
    """Compute the regularized negative log-likelihood

    Parameters
    ----------
    mu_fit : array_like
        The current fit positions.
        Shape: [n_pos, 2]
    mu : array_like
        The estimated fit positions by the neural network.
        Shape: [n_pos, 2]
    sigma : array_like
        The estimate uncertainty of the asymmetric Gaussian.
        Shape: [n_pos, 1]
    r : array_like
        The estimate asymmetry parameter of the asymmetric Gaussian.
        Shape: [n_pos, 1]
    t : array_like
        The time stamps of each position.
        Shape: [n_pos, 1]
    use_kde : bool, optional
        Whether or not to use KDE as regularization.
        --> Not implemented yet!
    kdes : list of KDEs, optional
        A list of KDEs to use.
    splines : list of Splines, optional
        A list of splines to use.

    Returns
    -------
    float
        The scalar regularized negative log-likelihood.
    """
    mu_fit = np.reshape(mu_fit, [-1, 2])

    # compute velocities and accelerations
    vel_abs, acc_abs, acc_parallel, acc_perp = get_vel_and_acc(mu_fit, t)
    acc_perp_abs = np.abs(acc_perp)

    # set up likelihood
    neg_llh = np.sum(-basis_functions.log_asymmetric_gauss(
        mu_fit, mu=mu, sigma=sigma, r=r))
    #neg_llh = np.sum(-basis_functions.log_gauss(
    #    mu_fit, mu=mu, sigma=sigma))

    if use_kde:
        kde_data = np.concatenate(
            [
                #acc_perp_abs,
                #acc_parallel,
                vel_abs,
            ],
            axis=1,
        )
        reg_loss = -kdes[1].score(kde_data)
        loss = neg_llh + reg_loss
    else:
        # # add regularization: acceleration should be small
        # # start penalizing above 0.8 m/s^2 (at normal rate)
        # # start penalizing above 0.3 m/s^2 (at 3s/readout)
        # # ToDo: parameterize this f(acc, dt) --> loss
        # dt = np.zeros_like(t)
        # dt[1:] = np.diff(t, axis=0)
        # reg_acc_1th = np.where(
        #     acc_abs > 0.8,  # m / s^2
        #     np.exp(0 + 2*acc_abs),
        #     np.exp(1.2 + 0.5*acc_abs),
        # )
        # reg_acc_13th = np.where(
        #     acc_abs > 0.3,  # m / s^2
        #     np.exp(0 + 3*acc_abs),
        #     np.exp(0.75 + 0.5*acc_abs),
        # )
        # reg_acc = np.sum(np.where(
        #     dt > 1.5,  # s / readout
        #     reg_acc_13th,
        #     reg_acc_1th,
        # ))
        # #reg_acc = np.sum(np.exp(1 + 2*acc))

        # compute delta t
        dt = np.zeros_like(t)
        dt[1:] = np.diff(t, axis=0)

        # get regularization indices for specified dt
        indices = get_reg_param_index(dt[:, 0])

        reg_acc = get_acc_regularization(
            indices, acc_abs, acc_parallel, acc_perp,
            add_parallel_reg=False,
            add_perp_reg=True,
            add_abs_reg=True,
        )

        # add regularization: velocity should be small
        # anything above 1.015 m/s should not exist
        # This is pretty independent of the time deltas dt
        reg_vel = np.sum(reg_velocity_abs(vel_abs, max_cut_off=0.7177))

        if kdes is not None:
            reg_kde = 0.
            vel_abs_clipped = np.clip(vel_abs, 0., 0.72)
            for i in np.unique(indices):
                mask = indices == i
                reg_kde += -kdes[i].score(vel_abs_clipped[mask])

            # reg_kde = np.fromiter(map(lambda v: kdes[v[0]].score(v[1][:, None]),
            #                           zip(indices, vel_abs_clipped)),
            #                       dtype=float)
            # reg_kde = np.sum(reg_kde)
        else:
            reg_kde = 0.

        if splines is not None:
            reg_splines = 0.
            for i in np.unique(indices):
                mask = indices == i
                reg_splines += np.sum(-np.log(
                    np.clip(splines[i](vel_abs[mask]), 1e-4, 1e4)))

        else:
            reg_splines = 0.

        loss = neg_llh + reg_acc + reg_vel + reg_kde + reg_splines

    return loss


def get_acc_regularization(indices, acc_abs, acc_parallel, acc_perp,
                           add_parallel_reg=True, add_perp_reg=True,
                           add_abs_reg=True):
    """Get Regularization for Acceleration vector components.

    Parameters
    ----------
    indices : array_like
        The sampling frequency indices as obtained via get_reg_param_index.
        Shape: [n_pos, 1]
    acc_abs : array_like
        The magnitude of the acceleration vector.
        Shape: [n_pos, 1]
    acc_parallel : array_like
        The signed magnitude of the parrallel acceleration component.
        Shape: [n_pos, 1]
    acc_perp : array_like
        The signed magnitude of the perpendicular acceleration component.
        Shape: [n_pos, 1]

    Returns
    -------
    float
        The scalar regularization loss.
    """

    # compute loss for absolute perpendicular component
    acc_perp_abs = np.abs(acc_perp)

    # get regularization parameter for specified dt
    params_abs = reg_params_acc_abs[indices][..., None]
    params_perp = reg_params_acc_perp_abs[indices][..., None]
    params_parallel = reg_params_acc_parallel_abs[indices][..., None]

    reg_loss = 0.

    # compute loss for magnitude of acc vector
    if add_abs_reg:
        reg_loss += np.sum(reg_acc_abs(
            acc_abs,
            max_cut_off=params_abs[:, 0],
            reg_hard=params_abs[:, 1],
            reg_soft=params_abs[:, 2],
        ))

    # compute loss for absolute perpendicular component
    if add_perp_reg:
        reg_loss += np.sum(reg_acc_abs(
            acc_perp_abs,
            max_cut_off=params_perp[:, 0],
            reg_hard=params_perp[:, 1],
            reg_soft=params_perp[:, 2],
        ))

    # compute loss for parallel component
    if add_parallel_reg:
        reg_loss += np.sum(np.where(
            acc_parallel > 0,
            reg_acc_abs(
                acc_parallel,
                max_cut_off=params_parallel[:, 1],
                reg_hard=params_parallel[:, 2],
                reg_soft=params_parallel[:, 3],
            ),
            reg_acc_abs(
                -acc_parallel,
                max_cut_off=params_parallel[:, 0],
                reg_hard=params_parallel[:, 2],
                reg_soft=params_parallel[:, 3],
            )
        ))
    return reg_loss


def get_reg_param_index(dt, max_index=12):
    """Get the regularization parameter index for given delta ts.

    Parameters
    ----------
    dt : array_like
        The time between samples. The first delta time should be zero.
        Shape: [n_pos, 1]
    max_index : int, optional
        The maximum available parameter index.

    Returns
    -------
    array_like
        An array of regularization parameter indices. These can be used
        to obtain the regularization parameters for given delta ts.
    """
    time_per_nth = 0.23  # s
    index = np.floor(dt / time_per_nth - 0.5).astype(int)
    index = np.clip(index, 0, max_index)
    return index


def reg_acc_abs(acc_abs, max_cut_off=0.8, reg_hard=2., reg_soft=0.5):
    """Return regularization loss for the magnitude of an acceleration.

    Parameters
    ----------
    acc_abs : array_like
        The magnitude of the acceleration vector or the absolute component
        of the perpendicular acceleration vector.
        Shape: [n_pos, 1]
    max_cut_off : float or array_like, optional
        The assumed maximum value that should exist. Anything above this
        is regularized very hard.
        Shape: [n_pos, 1]
    reg_hard : float or array_like, optional
        The regularization exp factor for values exceeding `max_cut_off`.
        Shape: [n_pos, 1]
    reg_soft : float or array_like, optional
        The regularization exp factor for values under `max_cut_off`.
        Shape: [n_pos, 1]

    Returns
    -------
    array_like
        The regularization loss for each term.
        Shape: [n_pos, 1]
    """
    offset = max_cut_off * (reg_hard - reg_soft)
    reg = np.where(
        acc_abs > max_cut_off,  # m / s^2
        np.exp(0 + reg_hard*acc_abs),
        np.exp(offset + reg_soft*acc_abs),
    ) - np.exp(offset)
    return reg


def reg_velocity_abs(vel_abs, max_cut_off=0.7177):
    """Return regularization loss for the magnitude of the velocity.

    Parameters
    ----------
    vel_abs : array_like
        The magnitude of the velocity vector.
        Shape: [n_pos, 1]
    max_cut_off : float, optional
        The assumed maximum velocity that should exist. Anything above this
        is regularized.

    Returns
    -------
    array_like
        The regularization loss for each term.
        Shape: [n_pos, 1]
    """
    reg = np.where(
        vel_abs > max_cut_off,
        np.exp(10*(vel_abs - max_cut_off)),
        1.,
        # 1.-(1000 * (0.35 - vel_abs)**2),
        # 1. + 0.01 * (- 1./(10*vel_abs + 1e-1)
                    # - 1./(10*(max_cut_off - vel_abs) + 1e-1)),
    )
    return reg


def postprocess(mu, sigma, r, t, x0=None, use_kde=False,
                kdes=None, splines=None,
                pre_fitting_size=None, verbose=True):
    """Postprocess estimated positions.

    Run regularized MLE to estimate regularized positions. This can utilize
    the connection between points and the fact that the roboter should move
    continously.

    Parameters
    ----------
    mu : array_like
        The estimated fit positions by the neural network.
        Shape: [n_pos, 2]
    sigma : array_like
        The estimate uncertainty of the asymmetric Gaussian.
        Shape: [n_pos, 1]
    r : array_like
        The estimate asymmetry parameter of the asymmetric Gaussian.
        Shape: [n_pos, 1]
    t : array_like
        The time stamps of each position.
        Shape: [n_pos, 1]
    x0 : array_like, optional
        An initial guess for the best fit positions. If none is provided, the
        provided `mu` will be used.
        Shape: [n_pos, 2]
    use_kde : bool, optional
        Wheter or not to use a kde for the regularization.
        --> Not implemented yet!
    kdes : list of KDEs, optional
        A list of KDEs to use.
    splines : list of splines, optional
        List of spline functions to use.
    pre_fitting_size : int, optional
        If provided, a smaller chunk of the given size will be optimized first.
        These are then combined and then everything is optimized at once.
    verbose : bool, optional
        Print out result object.

    Returns
    -------
    array_like
        The updated and post-processed position estimates.
        Shape: [n_pos, 2]
    """
    if pre_fitting_size is not None:
        # split up mu in smaller chunks of maximum size
        # `pre_fitting_size` and fit these first
        # This could speed up overall convergence
        mu_list = []
        n_splits = np.ceil(len(mu) / pre_fitting_size)
        for indices in tqdm(np.array_split(np.arange(len(mu)), n_splits)):
            mu_split = mu[indices]
            sigma_split = sigma[indices]
            t_split = t[indices]
            r_split = r[indices]
            if x0 is not None:
                x0_split = x0[indices]
            else:
                x0_split = None
            mu_list.append(
                postprocess(
                    mu_split, sigma_split,
                    r_split, t_split,
                    x0=x0_split,
                    use_kde=use_kde,
                    verbose=False,
                )
            )
        x0 = np.concatenate(mu_list, axis=0)

    elif x0 is None:
        x0 = mu

    result = minimize(
        fun=likelihood,
        x0=x0.flatten(),
        args=(mu, sigma, r, t, use_kde, kdes, splines),
        #method='Nelder-Mead',
        method='L-BFGS-B',
        options={
            'ftol': 1e-7,
            'gtol': 1e-05,
            'maxfun': 10000000,
            'maxiter': 10000000,
        }
    )
    if verbose:
        print(result)
    return np.reshape(result.x, [-1, 2])
