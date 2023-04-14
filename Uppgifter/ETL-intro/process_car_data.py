import numpy as np
import os

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
speed_measurement_path = CURR_DIR_PATH + "\Docs\speed_measurement.txt"
speed_data = np.genfromtxt(speed_measurement_path, delimiter=",")

print(speed_measurement_path)