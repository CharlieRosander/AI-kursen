from forecast_ETL import ForecastETL


if __name__ == '__main__':
    api_call = ForecastETL()
    api_call.transform_forecast()
    api_call.save_files()
    # api_call.plot_forecast()
    api_call.init_db()
    print(api_call.harmonized_dataframe)
