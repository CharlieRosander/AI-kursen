import requests
import os
from dotenv import load_dotenv
import pandas as pd


class WeatherETL:
    def __init__(self, city_query):
        self.city_query = city_query
        self.raw_data = None
        self.normalized_dataframe = None
        self.harmonized_dataframe = None

    @staticmethod
    def get_api_key():
        load_dotenv()
        openweathermap_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        return openweathermap_api_key

    def extract_weather(self):
        url = f'http://api.openweathermap.org/data/2.5/weather?q=' \
              f'{self.city_query}&appid={self.get_api_key()}&units=metric'
        response = requests.get(url).json()
        self.raw_data = response
        return response

    def transform_weather(self):
        temperature = self.extract_weather()['main']['temp']
        description = self.extract_weather()['weather'][0]['description']
        country = self.extract_weather()['sys']['country']
        location = self.extract_weather()['name']
        feels_like = self.extract_weather()['main']['feels_like']
        data = {'Temperature': [temperature], 'Description': [description], 'Country': [country],
                'Location': [location],
                'Feels Like': [feels_like]}

        self.normalized_dataframe = pd.json_normalize(self.extract_weather())
        self.harmonized_dataframe = pd.DataFrame(data)
        return self.normalized_dataframe, self.harmonized_dataframe

    # save normalized and harmonized dataframe to csv
    def load_weather(self):
        self.normalized_dataframe.to_csv('Docs/normalized_weather.csv', index=False)
        self.harmonized_dataframe.to_csv('Docs/harmonized_weather.csv', index=False)


if __name__ == '__main__':
    api_call = WeatherETL("stockholm, se")
    api_call.transform_weather()
    # print(api_call.transform_weather())
    print(api_call.city_query)
    # print(api_call.raw_data)
    # print(api_call.normalized_dataframe)
    print(api_call.harmonized_dataframe)

    api_call2 = WeatherETL("london, uk")
    api_call2.transform_weather()
    # print(api_call2.transform_weather())
    print(api_call2.city_query)
    # print(api_call2.raw_data)
    # print(api_call2.normalized_dataframe)
    print(api_call2.harmonized_dataframe)

    api_call.load_weather()