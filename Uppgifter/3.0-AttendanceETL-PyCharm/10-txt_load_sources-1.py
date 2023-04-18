# The code presums a postgres connection on localhost
# Uses a config.ini file to hide db password

import os
import pandas as pd

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Data processing
speed_data_path = CURR_DIR_PATH + "/speed_measurements"

speed_data = pd.read_csv(
    speed_data_path + ".txt",
    sep=",",
    header=None,
    encoding="unicode_escape"
)

speed_data.columns = ["speed", "lisence_plate", "color", "time"]

#Write to new sources
speed_data.to_csv(speed_data_path + ".csv", index=False)
speed_data.to_json(speed_data_path + ".json")
speed_data.to_html(speed_data_path + ".html")
