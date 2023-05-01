import requests
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from datetime import datetime
import json


def get_api_key():
    load_dotenv()
    openweathermap_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    return openweathermap_api_key


city_query = "stockholm, se"
url = f'http://api.openweathermap.org/data/2.5/forecast?q={city_query}&appid={get_api_key()}&units=metric'

response = requests.get(url).json()
# create raw_data.json file in docs folder
with open('Docs/raw_data_json.json', 'w') as outfile:
    json.dump(response, outfile)

harmonized_df = pd.DataFrame(
    columns=['Fetch Date', 'Location', 'Date', 'Time', 'Temperature', 'Air Pressure', 'Weather Description',
             'Precipitation'])

# data = []
# for i in range(0, len(response["list"])):
#     date_time = response["list"][i]["dt_txt"]
#     date, time = date_time.split(' ')
#     temperature = response["list"][i]["main"]["temp"]
#     air_pressure = response["list"][i]["main"]["pressure"]
#     weather_description = response["list"][i]["weather"][0]["description"]
#     precipitation = response["list"][i]["pop"]
#     data.append({'Date': date, 'Time': time, 'Temperature': temperature, 'Air Pressure': air_pressure,
#                  'Weather Description': weather_description, 'Precipitation': precipitation})

# Dictionary instead of list
data = {}
for i in range(0, len(response["list"])):
    fetch_date = datetime.now().strftime("%Y-%m-%d")
    date_time = response["list"][i]["dt_txt"]
    date, time = date_time.split(' ')
    location = response["city"]["name"]
    temperature = response["list"][i]["main"]["temp"]
    air_pressure = response["list"][i]["main"]["pressure"]
    weather_description = response["list"][i]["weather"][0]["description"]
    precipitation = response["list"][i]["pop"]
    data[i] = {'Fetch Date': fetch_date, 'Date': date, 'Time': time, 'Location': location, 'Temperature': temperature,
               'Air Pressure': air_pressure,
               'Weather Description': weather_description, 'Precipitation': precipitation}

harmonized_df = pd.concat([harmonized_df, pd.DataFrame(data)])
harmonized_df.to_csv('Docs/harmonized_weather.csv', index=False)

# Create a postgres database and store the data in a table
# Create a connection to the database
connection = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="kaliber",
    port=5432)

# Create a cursor
cursor = connection.cursor()

# Create a table
cursor.execute("DROP TABLE IF EXISTS weather")
cursor.execute("CREATE TABLE IF NOT EXISTS weather "
               "(fetch_date text,"
               "date text, "
               "time text, "
               "location text, "
               "temperature real, "
               "air_pressure real,"
               "weather_description text, "
               "precipitation real)")

# Insert the data into the table
for i in range(0, len(response["list"])):
    fetch_date = datetime.now().strftime("%Y-%m-%d")
    date_time = response["list"][i]["dt_txt"]
    date, time = date_time.split(' ')
    location = response["city"]["name"]
    temperature = response["list"][i]["main"]["temp"]
    air_pressure = response["list"][i]["main"]["pressure"]
    weather_description = response["list"][i]["weather"][0]["description"]
    precipitation = response["list"][i]["pop"]
    cursor.execute("INSERT INTO weather VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                   (fetch_date, date, time, location, temperature, air_pressure, weather_description, precipitation))

# Commit the changes
connection.commit()

# Close the cursor and the connection
cursor.close()
connection.close()

print(response.keys())
print(response["city"].keys())
print(response["list"][0].keys())
# print(response["list"][0]["main"].keys())
# print(response["list"][0]["weather"][0].keys())
# print(response["list"][0]["weather"][0]["description"])
# print(response["list"][0]["wind"].keys())
