# The code presums a postgres connection on localhost
# Uses a config.ini file to hide db password

import os
import pandas as pd
import psycopg2 as ps  # postgres sql
import configparser  # package to fetch external config files

from sqlalchemy import create_engine  # sql driver

config = configparser.ConfigParser()  # instansiation of configs
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# read config file (mainly to keep pw safe)
config.read(CURR_DIR_PATH + "/config.ini")
db_pw = config.get("DEV", "psycopg2_pw")  # Load config pw for dev env

def postgres_creator():  # sqlalchemy requires callback-function for connections
  return ps.connect(
      dbname="speed_measurements",  # name of table
      user="postgres",
      password=db_pw,
      host="localhost"
  )

postgres_engine = create_engine(
    url="postgresql+psycopg2://localhost",  # driver identification + dbms api
    creator=postgres_creator  # connection details
)

sqlite_engine = create_engine("sqlite:///" + CURR_DIR_PATH + "/database.db") # local sqlite


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
speed_data.to_sql("speed_measurements", postgres_engine, if_exists="replace") # write to postgres
speed_data.to_sql("speed_measurements", sqlite_engine, if_exists="replace") # write to sqlite
