Overview
========

This project is aimed at solving the TU Dortmund summer-school-2020 Hackathon.
For this purpose multiple modules, python notebooks, and scripts were created.
An overview of these is provided below.

InvestigateData.ipynb
---------------------

General jupyter notebook for prototyping and data investigation

modules directory
-----------------

Directory with helper functions.
This can in principle be used as a python package.
ToDo: properly create a python package.

run_competition.py
------------------

This script is used to run part 2 of the competition, e.g. the live
robot controlling.

collect_sensor_data.py
----------------------

This script collects data from the REST API to obtain an averaged
sensor activation. From these the sensor normalization is computed.
An alternative approach is to use the training data for this.
Ideally, results of both approaches should match up.

monitor_localization.py
-----------------------

This script displays and monitors the current robot position.
![An example screenshot of the monitoring script.](https://github.com/mhuen/REAML2020Hackathon/blob/master/example_view.png?raw=true)
