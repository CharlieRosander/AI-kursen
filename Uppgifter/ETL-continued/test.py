import pandas as pd
import os

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR_PATH, "data")


data_dict = {}

for file in os.listdir(DATA_DIR):
    class_name = file.partition("_")[0]

    if not file.endswith((".csv", ".txt")):
        continue

    file_path = os.path.join(DATA_DIR, file)
    data = pd.read_csv(file_path)

    if class_name in data_dict:
        data_dict[class_name] = pd.concat([data_dict[class_name], data])
    else:
        data_dict[class_name] = data

for class_name, class_data in data_dict.items():
    output_filename = f"{class_name}_data.csv"
    output_filepath = os.path.join(CURR_DIR_PATH, output_filename)
    class_data.to_csv(output_filepath, index=False)

data_df = pd.concat(data_dict.values(), axis=0)
data_df["class_name"] = [key for key in data_dict.keys()
                         for _ in range(len(data_dict[key]))]
data_df.reset_index(inplace=True)
data_df.drop("index", axis=1, inplace=True)

if "firstname" in data_df.columns and "surname" in data_df.columns:
    data_df["full_name"] = f"{data_df['firstname']} {data_df['surname']}"
    data_df.drop(["firstname", "surname"], axis=1, inplace=True)

elif "name" in data_df.columns:
    data_df.rename(columns={"name": "full_name"}, inplace=True)

if "attendance" in data_df.columns:
    data_df.rename(columns={"attendance": "late"}, inplace=True)

print(data_df)
data_df.to_csv("data.csv", index=False)
