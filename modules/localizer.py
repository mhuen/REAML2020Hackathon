"""
Robot Controller class based on the example code provided
for the TU Dortmund summer-school-2020 Hackathon.
"""
from __future__ import print_function, division

import requests
import json
import time
import numpy as np
import pandas as pd
import multiprocessing

from modules.data import get_data_from_data_frame
from modules.model import ModelWrapper


class Localizer:
    def __init__(
            self,
            model_file,
            list_size=20,
            server='http://s17.phynetlab.com/',
            allowed_secret='5ebe2294ecd0e0f08eab7690d2a6ee69',
            ):
        """Class that performs asynchronous localization

        Parameters
        ----------
        model_file : str
            The path to the NN model that will be used.
        list_size : int, optional
            The maximum size of the list.
        server : str, optional
            The url to the server.
        allowed_secret : str, optional
            The secret to acces the API.
        """
        self.url_data = server + allowed_secret + '/current_values'

        self.list_size = list_size
        self.model_file = model_file

        # create a multiprocessing manager and shared lists
        self.manager = multiprocessing.Manager()

        # keep track of the predictions
        self.predictions = self.manager.list()

        # create a list of processes to keep track of them
        self.processes = []

        # start background process
        self.start_localization_process()

    def start_localization_process(self):
        """Start multiprocessing localization process

        This asynchronously collects sensor data from the REST API and performs
        predictions. A list of the last `list_size` data points is kept.
        """

        def update_predictions():
            """Update shared list of predictions asynchronously

            This process updates the shared list of predictions by:
                1) Query data base to obtain new data frame
                2) Run prediction on this data frame (without post-processing)
                3) Run post-processing on all predictions in shared list
                4) Update shared list and current position
            """

            # Load model and create ModelWrapper object
            # This needs to be done in this process due to issues with
            # multi-processing and keras, where the prediction would hang
            # indefenitely
            print('Creating Model Wrapper')
            model_wrapper = ModelWrapper(self.model_file)

            # create lists for mu, sigma, r, t
            # These lists are only available to the process running this
            mu_list = []
            sigma_list = []
            r_list = []
            t_list = []
            pos_list = []

            last_time = -float('inf')
            while True:

                # ----------------------------------------------------
                # 1) get new data frame from the collected sensor data
                # ----------------------------------------------------
                found_new_data = False
                while not found_new_data:

                    # get current data frame
                    try:
                        df_data = self.get_current_data_frame()
                    except Exception as e:
                        print(e)
                        continue

                    # convert to table-structure and extract average time
                    X_i, t_i = get_data_from_data_frame(df_data)

                    # check if data is new
                    if np.abs(t_i - last_time) > 0.2:
                        last_time = t_i
                        found_new_data = True
                    else:
                        # sleep for a little to not over do it with requests
                        time.sleep(0.5)

                # extend dimension to account for batch size
                X_i = X_i[None, ...]
                t_i = np.reshape(t_i, [1, 1])

                # -----------------------------------
                # 2) Run prediction on new data frame
                # -----------------------------------
                # Now that we have new data, we need to run the prediction
                # on this data frame
                mu_i, sigma_i, r_i = model_wrapper.get_posterior(
                    X_i, check_data=False)

                # add new element to lists
                t_list.append(t_i)
                mu_list.append(mu_i)
                sigma_list.append(sigma_i)
                r_list.append(r_i)

                # for an intial guess we use the model prediction
                pos_list.append(mu_i)

                # if the length is too long, remove the first element
                if len(mu_list) > self.list_size:
                    pos_list.pop(0)
                    t_list.pop(0)
                    mu_list.pop(0)
                    sigma_list.pop(0)
                    r_list.pop(0)

                # sanity check
                assert len(mu_list) == len(t_list), 'Length mis-match!'
                assert len(mu_list) == len(r_list), 'Length mis-match!'
                assert len(mu_list) == len(sigma_list), 'Length mis-match!'
                assert len(mu_list) == len(pos_list), 'Length mis-match!'

                # ----------------------
                # 3) Run post-processing
                # ----------------------
                if len(mu_list) > 1:
                    # collect data from the lists
                    pos = np.concatenate(pos_list, axis=0)
                    t = np.concatenate(t_list, axis=0)
                    mu = np.concatenate(mu_list, axis=0)
                    sigma = np.concatenate(sigma_list, axis=0)
                    r = np.concatenate(r_list, axis=0)

                    # run post-processing
                    pos_new = model_wrapper.post_process(
                        mu, sigma, r, t, x0=pos, verbose=False)

                    # update pos_list
                    pos_list = [pos[None, ...] for pos in pos_new]

                # ----------------------------
                # 4) Update public shared list
                # ----------------------------
                self.predictions.append((t_list[-1], pos_list[-1]))

        # create background process
        print('Starting localization process')
        process = multiprocessing.Process(target=update_predictions)
        process.daemon = True
        process.start()
        self.processes.append(process)
        print('Successfully started localization process')

    def localize(self):
        """Localize robot.

        Returns
        -------
        float
            The time of the last prediction
        array_like
            The predicted position of the last prediction.
            Shape: [2]
        """
        while len(self.predictions) == 0:
            time.sleep(0.2)

        last_prediction = self.predictions[-1]
        return last_prediction[0][0, 0], last_prediction[1][0]

    def get_current_data_frame(self):
        """Get Current sensor values as a pandas dataframe
        """

        # read the current values from the APIs
        response = requests.request("GET", self.url_data)
        df_frame = pd.read_json(response.text.encode('utf8'))
        return df_frame

    def kill(self):
        """Kill Multiprocessing queues and workers
        """
        for process in self.processes:
            process.terminate()

        time.sleep(0.1)
        for process in self.processes:
            process.join(timeout=1.0)

        self.manager = None

    def __del__(self):
        self.kill()
