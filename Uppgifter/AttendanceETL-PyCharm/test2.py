import pandas as pd
import os

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR_PATH, "data")

data_dict = {}

# Reads all csv/txt files in the data directory and stores them in a dictionary with the class name as the key
# and the data as the value
for file in os.listdir(DATA_DIR):
    class_name = file.partition("_")[0]

    file_path = os.path.join(DATA_DIR, file)
    data = pd.read_csv(file_path)
    data = data.rename(columns={"Unnamed: 0": "student_id"})

    if class_name in data_dict:
        data_dict[class_name] = pd.concat([data_dict[class_name], data])
    else:
        data_dict[class_name] = data

# Converts the dictionary to a dataframe
data_df = pd.concat(data_dict.values(), axis=0)

# Checks if "firstname" and "surname" columns exist and combines them into a single column "full_name"
if "firstname" in data_df.columns and "surname" in data_df.columns:
    data_df["full_name"] = data_df["firstname"] + " " + data_df["surname"]
    data_df.drop(["firstname", "surname"], axis=1, inplace=True)

# Checks if "name" column exists and moves the values from "name" to "full_name"
# and drops the "name" column
if "name" in data_df.columns:
    data_df["full_name"] = data_df["name"].fillna(data_df["full_name"])
    data_df.drop("name", axis=1, inplace=True)

# Checks if "attendance" column exists and moves the values from "attendance" to "late"
# and drops the "attendance" column
if "attendance" in data_df.columns:
    data_df["late"] = data_df["attendance"].fillna(data_df["late"])
    data_df.drop("attendance", axis=1, inplace=True)


# Saves each subject to a separate CSV file
for subject in data_df["subject"].unique():
    subject_data = data_df.loc[data_df["subject"] == subject]
    output_filename = f"{subject}_data.csv"
    output_filepath = os.path.join(CURR_DIR_PATH, output_filename)
    subject_data.to_csv(output_filepath, index=False)

# combine data of all students that have < 60 attendance from the "late" column
# and save the combined data to a CSV file
late_data = data_df.loc[data_df["late"] < 60]
output_filename = "absence_june.csv"
output_filepath = os.path.join(CURR_DIR_PATH, output_filename)
late_data.sort_values(by=["late"], ascending=False, inplace=True) # FIX THIS
late_data.to_csv(output_filepath, index=False)

