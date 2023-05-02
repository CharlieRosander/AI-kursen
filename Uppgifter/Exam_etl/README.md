# ForecastETL

This is a weather forecast ETL (Extract, Transform, Load) Python script that fetches weather forecast data from the OpenWeatherMap API, processes the data, and stores it in a PostgreSQL database. The script also generates a line plot of the forecasted temperatures.
# Prerequisites
NOTE: I've had some issues with psycopg2 and noticed that you might have to install **psycopg2** outside the venv to get it working

    Python 3.6 or higher
    pandas
    requests
    psycopg2
    matplotlib
    python-dotenv

To install the required packages, run:

    pip install pandas requests psycopg2 matplotlib python-dotenv

OR if you have the requirements.txt

    pip install -r requirements.txt

## Setup

Set up a PostgreSQL database and provide the password in the .env file:
    
    ```
    DB_PASSWORD=<your_postgresql_password>
    ```

Sign up for a free account at OpenWeatherMap and obtain an API key.
Create a .env file in the root directory of the project and add the following lines:

    OPENWEATHERMAP_API_KEY=<your_api_key>
    DB_PASSWORD=<your_postgresql_password>

Replace <your_api_key> with the API key you received from OpenWeatherMap.




The database model used in this program is the default postgresql database that is created when you first install postgres,
If you want you can change the db model in the forecast_etl.py on these lines:

    ```
    connection = psycopg2.connect(
            host="localhost",
            database="postgres",
            user="postgres",
            password=db_password,
            port=5432)
    ```


Instantiate the ForecastETL class:

    forecast_etl = ForecastETL()


Extract the forecast data:

    forecast_etl.extract_forecast()


Transform the forecast data into two DataFrames (normalized and harmonized):

    normalized_dataframe, harmonized_dataframe = forecast_etl.transform_forecast()

Save the forecast data to files:

    forecast_etl.save_files()

Initialize the PostgreSQL database and create the required tables:

    forecast_etl.init_db()

Load the data into the PostgreSQL database:


    forecast_etl.load_db(connection, cursor)

Plot the forecast data:

    forecast_etl.plot_forecast()

## Output

The script generates the following output files:

    Docs/raw_data_json.json: Raw JSON data from the OpenWeatherMap API response
    Docs/normalized_dataframe_json.json: Normalized JSON data
    Docs/harmonized_forecast.csv: Harmonized CSV data

A line plot of the forecasted temperatures will be displayed in a separate window.

