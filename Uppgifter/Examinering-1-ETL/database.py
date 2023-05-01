import psycopg2
from datetime import datetime


def init_db(self):
    # Connect to the database if it exists, otherwise create a new one
    connection = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="kaliber",
        port=5432)

    # Create a cursor
    cursor = connection.cursor()

    # Create the date dimension table
    cursor.execute("DROP TABLE IF EXISTS dim_date")
    cursor.execute("CREATE TABLE dim_date "
                   "(year INTEGER, "
                   "month INTEGER, "
                   "day INTEGER, "
                   "full_date DATE PRIMARY KEY)")

    # Create the time dimension table
    cursor.execute("DROP TABLE IF EXISTS dim_time")
    cursor.execute("CREATE TABLE dim_time "
                   "(hour INTEGER, "
                   "minute INTEGER, "
                   "full_time TIME PRIMARY KEY)")

    # Create the location dimension table
    cursor.execute("DROP TABLE IF EXISTS dim_location")
    cursor.execute("CREATE TABLE dim_location "
                   "(latitude REAL, "
                   "longitude REAL, "
                   "location_name TEXT PRIMARY KEY)")

    # Create the weather table with foreign keys to the dimension tables
    cursor.execute("DROP TABLE IF EXISTS weather")
    cursor.execute("CREATE TABLE weather "
                   "(fetch_date TEXT, "
                   "date_id DATE REFERENCES dim_date(full_date), "
                   "time_id TIME REFERENCES dim_time(full_time), "
                   "location_id TEXT REFERENCES dim_location(location_name), "
                   "temperature REAL, "
                   "air_pressure REAL, "
                   "weather_description TEXT, "
                   "precipitation REAL)")

    # Load the data into the table
    self.load_db(connection, cursor)

    # Close the cursor and the connection
    cursor.close()
    connection.close()
