import pandas as pd
import os

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR_PATH, "data")


class ETLData:
    def __init__(self):
        self.data_dict = {}

    # Reads all csv/txt files in the data directory and stores them in a dictionary with the class name as the key
    # and the data as the value
    def extract_files(self):
        for file in os.listdir(DATA_DIR):
            class_name = file.partition("_")[0]

            file_path = os.path.join(DATA_DIR, file)
            class_data = pd.read_csv(file_path)

            class_data = class_data.rename(
                columns={"Unnamed: 0": "student_id"})

            if class_name in self.data_dict:
                self.data_dict[class_name] = pd.concat(
                    [self.data_dict[class_name], class_data])
            else:
                self.data_dict[class_name] = class_data
        return self.data_dict

    def transform_data(self):
        # Converts the dictionary to a dataframe
        data_df = pd.concat(self.data_dict.values(), axis=0)

        # Checks if "firstname" and "surname" columns exist and combines them into a single column "full_name"
        if "firstname" in data_df.columns and "surname" in data_df.columns:
            data_df["full_name"] = data_df["firstname"] + \
                " " + data_df["surname"]
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

        return data_df

    @staticmethod
    def load_data(data_df):
        # Combines all data that have the same "subject" data into their own csv file
        for subject in data_df["subject"].unique():
            subject_data = data_df.loc[data_df["subject"] == subject]
            output_filename = f"{subject}_data.csv"
            output_filepath = os.path.join(CURR_DIR_PATH, output_filename)
            subject_data.to_csv(output_filepath, index=False)

        # Combines all data into a combined single csv file
        data_df.to_csv("combined_data.csv", index=False)


if __name__ == "__main__":
    etl = ETLData()
    etl.extract_files()
    etl_transformed = etl.transform_data()
    etl.load_data(etl_transformed)
