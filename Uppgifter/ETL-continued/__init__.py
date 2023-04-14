import pandas as pd
import os

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
biology_data_path = CURR_DIR_PATH + "/data/biology_06.txt"

class_name = None

data_df = []
data_temp = []

for file in os.listdir(CURR_DIR_PATH + "/data"):
    filename = file.partition("_")[0]

    if file.partition("_")[0] == filename:
        data = pd.read_csv(CURR_DIR_PATH + "/data/" + file)
        data_temp.append(data)
        for data in data_temp:
            data_df.append(data)

        class_name = file.partition("_")[0] + "_data.csv"

    for data in data_temp:
        data_df.append(data)
        
    data_df = pd.concat(data_df)
    data_df.to_csv(class_name, index=False)
    data_temp.clear()
