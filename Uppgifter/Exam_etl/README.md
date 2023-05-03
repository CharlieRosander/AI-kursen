# 🌦️ ForecastETL

This is a weather forecast ETL (Extract, Transform, Load) Python script that fetches weather forecast data from the OpenWeatherMap API, processes the data, and stores it in a PostgreSQL database. The script also generates a line plot of the forecasted temperatures.

# 📚 Prerequisites

**NOTE**: I've had some issues with psycopg2 and noticed that you might have to install **psycopg2** outside the venv to get it working

--------------------------------
    - Python 3.6 or higher
    - pandas
    - requests
        - psycopg2
        OR
        - psycopg2-binary
    - matplotlib
    - python-dotenv
--------------------------------

    To install the required packages, run:

        pip install pandas requests psycopg2 matplotlib python-dotenv

    OR if you have the requirements.txt

        pip install -r requirements.txt

# 🔧 Setup
    Set up a PostgreSQL database and provide the password in the .env file:

    DB_PASSWORD=<your_postgresql_password>

    Sign up for a free account at OpenWeatherMap and obtain an API key.
    Create a .env file in the root directory of the project and add the following lines:

    OPENWEATHERMAP_API_KEY=<your_api_key>
    DB_PASSWORD=<your_postgresql_password>

    Replace <your_api_key> with the API key you received from OpenWeatherMap.

    Replace <your_postgresql_password> with the password you set for your PostgreSQL database.

    The database model used in this program is the default postgresql database
     that is created when you first install postgres

    If you want you can change the db model in the forecast_etl.py on these lines:

```py
        connection = psycopg2.connect(
                host="localhost",
                database="postgres",
                user="postgres",
                password=db_password,
                port=5432)
```

# 🏃‍♂️ Running the script
     To run the program, run main.py in your IDE or from the command line:

        python main.py


#### Instantiate the ForecastETL class:
    forecast_etl = ForecastETL()

#### Extract the forecast data:
    forecast_etl.extract_forecast()

#### Transform the forecast data into two DataFrames (normalized and harmonized):
    normalized_dataframe, harmonized_dataframe = forecast_etl.transform_forecast()

#### Save the forecast data to files:
    forecast_etl.save_files()

#### Initialize the PostgreSQL database and create the required tables:
    forecast_etl.init_db()

#### Load the data into the PostgreSQL database:
    forecast_etl.load_db(connection, cursor)

#### Plot the forecast data:
    forecast_etl.plot_forecast()

# 📁 Output
#### The script generates the following output files:

    Docs/raw_data_json.json: Raw JSON data from the OpenWeatherMap API response
    Docs/normalized_dataframe_json.json: Normalized JSON data
    Docs/harmonized_forecast.csv: Harmonized CSV data

    A line plot of the forecasted temperatures will be displayed in a separate window.

    The program uses PostGreSQL to store the data in tables and dimensional tables.

# 💡 Possible improvements & Thoughts
### Make the script more user-friendly:
    Right now you have to modify the code to get certain results. 
    For example the city request is explicitly set,
    this was done with a user input before but we could not make it work nicely with Airflow.

    Plotting the graph could also be optional, where you get an option to plot it or not.

### 🔄 Sharing the project
    As this project uses airflow which does not natively run on windows we had to use WSL,
    which makes the project "hard" to share as the user need to have the same setup to be able to run the DAG.

    Docker-compose was initially the preferred option but due to hardware-limitations 
    and time-constraints we had to go with WSL.

    Ideally we would also like to make the project into a docker image, 
    but we did not have time to see if this was a possibility.

### 🧩 Modularization:
    Separate the different aspects of the code into modules (e.g., database handling, 
    API requests, data transformation) for better organization and maintainability.

### 🔧 Configuration:
    Use a configuration file to store settings such as database credentials, API keys, 
    and city queries so that these can be easily updated without modifying the code.

### 🌐 Additional data sources:
    Integrate more data sources to enhance the quality and comprehensiveness of the weather forecast.

### 📈 Plotting:
    The plot can be improved for more readability and better visualization of the data.

### 🧑‍💻 Group-members

    Charlie Rosander
    Ronny Andersson
    Jacob Eriksson